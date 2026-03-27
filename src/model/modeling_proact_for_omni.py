import logging
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import transformers
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from torch.cuda.amp import autocast
import json

from peft import PeftModel
import torch.nn.functional as F
# from accelerate import infer_auto_device_map, dispatch_model
# from accelerate.utils import get_balanced_memory

from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerCausalLMOutputWithPast, apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from src.utils.constants import SILENT_TOKEN, ACTIVE_GEN_TOKEN, IGNORE_INDEX
from src.utils.utils import _maybe_hf, _is_local_dir
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor
from src.model.registry import build_model

from src.model import WrapQwen3VL, WrapQwen2_5VL, WrapQwen2VL
# from src.model import WrapQwen3VLProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

_MODEL_TAG_MAP = {
    "Qwen/Qwen2.5-Omni-7B": 'qwen2_5omni',
    "Qwen/Qwen3-VL-8B-Instruct": 'qwen3vl',
    "Qwen/Qwen2.5-VL-7B-Instruct": 'qwen2_5vl',
    "chenjoya/LiveCC-7B-Instruct": "qwen2vl",
    "/home/dyh/models--Qwen--Qwen2.5-Omni-7B/qwen/Qwen2.5-Omni-7B": 'qwen'
}

_MODEL_CLASS_MAP = {
    "Qwen/Qwen2.5-Omni-7B": Qwen2_5OmniForConditionalGeneration,
    # "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration,
    # "Qwen/Qwen3-VL-8B-Instruct": Qwen3_VLForConditionalGeneration,
    # "Qwen/Qwen3-VL-2B-Instruct"
    # "Qwen/Qwen3-Omni-30B-A3B-Instruct": 
}

class Qwen2_5OmniMLP(nn.Module):
    def __init__(self, hidden_size, down_size, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size / 4)
        self.down_size =down_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.down_size, bias=bias)
        self.act_fn = ACT2FN['gelu']

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class ProActConfig(PretrainedConfig):
    model_type = "proact_model"
    model_name_or_path: str = "Qwen/Qwen2.5-Omni-7B"
    enable_audio_output: bool = False
    state_threshold: float = 0.5
    loss_active_scale: float = 1.0
    finetune_strategy: str = 'none'

'''
Enable the vl model to be proactive.
The model should support the following functions:
For training:


For inference:
'''
class ProAct_OmniModel(PreTrainedModel, GenerationMixin):
    config_class = ProActConfig
    base_model_prefix = "proact_mllm"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: ProActConfig):
        super(ProAct_OmniModel, self).__init__(config)

        self.model_name_or_path = config.model_name_or_path
        self.model_tag = _MODEL_TAG_MAP.get(self.model_name_or_path, 'unknown_model')
        self.enable_audio_output = config.enable_audio_output
        
        self.active_layer_id = config.active_layer_id if hasattr(config, 'active_layer_id') else -1
        logger.info(f'Using active_layer_id: {self.active_layer_id}')
        self.finetune_strategy = config.finetune_strategy
        attn_implementation_options = [
            'eager',
            'sdpa',
            'flash_attention_2',
        ]
        self.attn_implementation = attn_implementation_options[2]
        # self.attn_implementation = 'eager'

        self.active_eos_token = None
        self.silence_eos_token = None
        if self.model_name_or_path == "Qwen/Qwen2.5-Omni-7B":
            self.llm = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation,
                low_cpu_mem_usage=True,
                enable_audio_output=self.enable_audio_output,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path
            )
            self.state_proj = Qwen2_5OmniMLP(self.llm.thinker.config.text_config.hidden_size, 1)
            self.active_eos_token = ''
            self.silence_eos_token = '<|SILENCE|>'


        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": ['<|SILENCE|>', '<|query_start|>', '<|query_end|>', '<|history_start|>', '<|history_end|>', '<|FLAG|>']
        })
        print(f"[OK] Added special tokens to tokenizer.")

        # response mechanism related configs
        self.state_threshold = config.state_threshold
        self.loss_active_scale = config.loss_active_scale
        self.chunk_flag = '<|FLAG|>'
        logger.info(f'Using active eos token: {self.active_eos_token}')
        logger.info(f'Using silence oes token: {self.silence_eos_token}')
        logger.info(f'Using chunk_flag: {self.chunk_flag}')
        # self.vocab_size = self.llm.config.thinker_config.text_config.vocab_size
        # self.ignore_index = -100

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        # 训练时关 cache
        if hasattr(self.llm.thinker, "config"):
            self.llm.thinker.config.use_cache = False
        # 开 GC
        if hasattr(self.llm.thinker, "gradient_checkpointing_enable"):
            try:
                self.llm.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            except TypeError:
                self.llm.thinker.gradient_checkpointing_enable()
        # LoRA + GC：确保输入需要梯度（LoRA在get_peft_model之后）
        if hasattr(self.llm.thinker, "enable_input_require_grads"):
            self.llm.thinker.enable_input_require_grads()


    def _print_trainable_params(self):        
        trainable_params, all_params = 0, 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"thinker trainable param: {name}, param shape: {param.shape}")
            all_params += param.numel()
        
        logger.info("thinker trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_params, 100 * trainable_params / all_params))
        print("thinker trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_params, 100 * trainable_params / all_params))

    @staticmethod
    def _freeze_parameters(module: nn.Module):
        """
        Freeze all parameters in the given module.
        """
        for name, param in module.named_parameters():
            param.requires_grad = False

    def set_threshold(self, threshold):
        if hasattr(self, 'state_threhold'):
            self.state_threhold = threshold
            logger.info(f'Set state_threhold to {self.state_threhold}')
        else:
            logger.warning('No state_threhold attribute to set.')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        active_labels: Optional[torch.LongTensor] = None,
        output_active_logits: Optional[bool] = True,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        if output_active_logits:
            # 如果要输出active_logits,需要拿到hidden_states进行计算，将其设置为True
            output_hidden_states = True
        # 假如active labels仅包含ignore label和0，说明当前全为silence，则不计算main loss
        if 'qwen2_5omni' in _MODEL_TAG_MAP[self.model_name_or_path]:
            output = self.llm.forward(
                input_ids=input_ids,
                input_features=input_features,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                rope_deltas=rope_deltas,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_audio_in_video=use_audio_in_video,
                cache_position=cache_position,
                video_second_per_grid=video_second_per_grid,
            )
        elif 'qwen2vl' in _MODEL_TAG_MAP[self.model_name_or_path]:
            output = self.llm.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                logits_to_keep=0
            )
        else:
            raise NotImplementedError(f"Model {self.model_name_or_path} not supported yet.")
        # 损失由两部分组成，main loss +active loss，如果active labels全为-100或0，则不计算main loss
 
        # 计算active logits
        if output_active_logits and output.hidden_states is not None:
        # select which layer's hidden state to use for state classification
            layer_id = self.active_layer_id
            # print(f'Computing active logits from layer {layer_id} hidden states.')
            last_hidden_state = output.hidden_states[layer_id]
            active_logits = self.llm.state_proj(last_hidden_state)
            output.active_logits = active_logits
            output['active_logits'] = active_logits

        # 清理不必要的输出
        if not output_hidden_states:
            output.hidden_states = None
        # 垃圾回收
        torch.cuda.empty_cache()
        
        return output
    
    # def generate(self, *args, **kwargs):
    #     return self.llm.thinker.generate(*args, **kwargs)

    def get_position_ids(self, **kwargs):
        if 'qwen2_5' in _MODEL_TAG_MAP[self.model_name_or_path]:
            input_ids = kwargs.get('input_ids', None)
            image_grid_thw = kwargs.get('image_grid_thw', None)
            video_grid_thw = kwargs.get('video_grid_thw', None)
            attention_mask = kwargs.get('attention_mask', None)
            use_audio_in_video = kwargs.get('use_audio_in_video', False)
            audio_feature_lengths = kwargs.get('audio_feature_lengths', None)
            video_second_per_grid = kwargs.get('video_second_per_grid', None)
            max_position_id = kwargs.get('max_position_id', 0)
            position_ids, rope_deltas = self.llm.thinker.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_feature_lengths,
                second_per_grids=video_second_per_grid,
            )
            
            position_ids = position_ids.add(max_position_id+1)
            return position_ids
        elif 'qwen2vl' in _MODEL_TAG_MAP[self.model_name_or_path]:
            input_ids = kwargs.get('input_ids', None)
            image_grid_thw = kwargs.get('image_grid_thw', None)
            video_grid_thw = kwargs.get('video_grid_thw', None)
            attention_mask = kwargs.get('attention_mask', None)
            max_position_id = kwargs.get('max_position_id', 0)
            position_ids, mrope_position_deltas = self.llm.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            position_ids = position_ids.add(max_position_id+1)
            return position_ids
        elif 'internvl' in _MODEL_TAG_MAP[self.model_name_or_path]:
            pass
        elif 'deepseek' in _MODEL_TAG_MAP[self.model_name_or_path]:
            pass
        else:
            raise NotImplementedError(f"Model {self.model_name_or_path} not supported yet.")


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        # 原始的qwen 2.5 omni将position_ids置为None,这里注释掉
        # model_inputs["position_ids"] = None
        # inplace modification
        # model_kwargs["cache_position"][-1:] + num_new_tokens
        position_ids.add_(1)
        model_inputs["position_ids"] = position_ids

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def save_llm(self, output_dir: str):
        # 保存 LoRA adapter 或 base 模型
        if isinstance(self.llm.thinker, PeftModel):
            self.llm.thinker.save_pretrained(output_dir)
        else:
            self.llm.thinker.save_pretrained(output_dir)

    def save_response_head(self, output_dir: str):
        # 保存 state_proj
        state_proj_path = os.path.join(output_dir, "state_proj.pt")
        torch.save(self.state_proj.state_dict(), state_proj_path)

        # 保存结构配置
        sp_cfg = {"type": self.state_proj.__class__.__name__}
        if isinstance(self.state_proj, nn.Linear):
            sp_cfg.update({
                "in_dim": self.state_proj.in_features,
                "out_dim": self.state_proj.out_features,
                "bias": self.state_proj.bias is not None
            })
        elif isinstance(self.state_proj, Qwen2_5OmniMLP):
            sp_cfg.update({
                "hidden_size": self.state_proj.hidden_size,
                "down_size": self.state_proj.down_size,
                "bias": self.state_proj.gate_proj.bias is not None
            })
        else:
            raise ValueError(f"Unsupported state_proj type: {type(self.state_proj)}")

        with open(os.path.join(output_dir, "state_proj_config.json"), "w") as f:
            json.dump(sp_cfg, f, indent=2)
    # -----------------
    # 保存方法
    # -----------------
    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        # 保存 processor
        self.processor.save_pretrained(output_dir)
        if self.finetune_strategy == 'strategy1':
            self.save_llm(output_dir)
        elif self.finetune_strategy == 'strategy2':
            self.save_response_head(output_dir)
        elif self.finetune_strategy == 'strategy3':
            self.save_llm(output_dir)
            self.save_response_head(output_dir)
        else:
            raise ValueError(f"Unknown finetune strategy: {self.finetune_strategy}")


    # -----------------
    # 加载方法
    # -----------------
    # FIXME 目前不支持分片加载
    @classmethod
    def from_pretrained(
        cls,
        config=None,
        adapter_dir: Optional[str] = None,          # 本地路径 or HF repo_id（如 "oaaoaa/ai_comp"）
        base_model_id: str = "Qwen/Qwen2.5-Omni-7B",
        revision: Optional[str] = None,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        offload_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        weight_dir_prefix: str = "",                # 例如 "final_1009"
    ):
        """
        adapter_dir: 本地文件夹 或 HF repo_id
        weight_dir_prefix: 适配器权重所在的“子目录”，例如 "final_1009"
        """
        # ---- 1) 构建模型骨架（或此处显式加载基座）----
        model = cls(config)

        # ---- 2) processor：同样支持 subfolder ----
        try:
            if _is_local_dir(adapter_dir):
                proc_path = os.path.join(adapter_dir, weight_dir_prefix) if weight_dir_prefix else adapter_dir
                model.processor = Qwen2_5OmniProcessor.from_pretrained(
                    proc_path, trust_remote_code=trust_remote_code
                )
            else:
                model.processor = Qwen2_5OmniProcessor.from_pretrained(
                    adapter_dir, revision=revision, subfolder=weight_dir_prefix or None, trust_remote_code=trust_remote_code
                )
            print("[OK] Loaded processor from adapter bundle.")
        except Exception:
            model.processor = Qwen2_5OmniProcessor.from_pretrained(
                base_model_id, trust_remote_code=trust_remote_code
            )
            print("[INFO] Fallback: loaded processor from base model.")

        # 如果需要在这里加载基座的话，取消注释：
        # from transformers import AutoModelForCausalLM
        # model.llm.thinker = AutoModelForCausalLM.from_pretrained(
        #     base_model_id, device_map=device_map, torch_dtype=dtype, trust_remote_code=trust_remote_code
        # )
        if 'strategy1' in adapter_dir or 'strategy3' in adapter_dir:
        # ---- 3) 挂载 LoRA：关键 → 对 HF 仓库用 subfolder；对本地用拼好的路径 ----
            try:
                model.llm.thinker.to(torch.float32)  # 合并前用 fp32 更稳
                if _is_local_dir(adapter_dir):
                    lora_path = os.path.join(adapter_dir, weight_dir_prefix) if weight_dir_prefix else adapter_dir
                    model.llm.thinker = PeftModel.from_pretrained(
                        model.llm.thinker,
                        lora_path,                      # 本地真实目录
                    )
                else:
                    # HF 仓库：repo_id + subfolder
                    model.llm.thinker = PeftModel.from_pretrained(
                        model.llm.thinker,
                        adapter_dir,                    # 仅 repo_id，如 "oaaoaa/ai_comp"
                        revision=revision,
                        subfolder=weight_dir_prefix or None
                    )
                model.llm.thinker = model.llm.thinker.merge_and_unload()
                print(f"[OK] Loaded & merged LoRA from {adapter_dir} (subfolder={weight_dir_prefix}).")
            except Exception as e:
                print(f"[WARN] Fail to load LoRA weights from {adapter_dir}: {e}")

            model.llm.thinker.to(dtype)

        # ---- 4) 读 state_proj 配置（支持本地/HF子目录），并兼容旧配置 ----
        if 'strategy2' in adapter_dir or 'strategy3' in adapter_dir:
            sp_cfg_path = _maybe_hf(
                adapter_dir,
                "state_proj_config.json",
                revision=revision,
                subfolder=weight_dir_prefix or None
            )
            with open(sp_cfg_path, "r", encoding="utf-8") as f:
                sp_cfg = json.load(f)

            # ---- 5) 恢复 state_proj（支持 Linear 与 Qwen2_5OmniMLP）----
            def _build_state_proj(sp_cfg: dict):
                sp_type = sp_cfg.get("type", "Linear")  # 向下兼容：老版本没有 type
                if sp_type in ("Linear", "nn.Linear"):
                    in_dim = sp_cfg["in_dim"]
                    out_dim = sp_cfg["out_dim"]
                    bias = sp_cfg.get("bias", True)
                    return nn.Linear(in_dim, out_dim, bias=bias)
                elif sp_type == "Qwen2_5OmniMLP":
                    # 需要你已有的类定义在可见作用域
                    hidden_size = sp_cfg["hidden_size"]
                    down_size = sp_cfg["down_size"]
                    bias = sp_cfg.get("bias", False)
                    return Qwen2_5OmniMLP(hidden_size=hidden_size, down_size=down_size, bias=bias)
                else:
                    raise ValueError(f"Unsupported state_proj type: {sp_type}")

            model.state_proj = _build_state_proj(sp_cfg)

            sp_w_path = _maybe_hf(
                adapter_dir,
                "state_proj.pt",
                revision=revision,
                subfolder=weight_dir_prefix or None
            )
            sd = torch.load(sp_w_path, map_location="cpu")

            # 容错：若权重与结构不完全对齐，给出更友好的报错
            try:
                missing, unexpected = model.state_proj.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"[WARN] state_proj load_state_dict: missing={missing}, unexpected={unexpected}")
            except RuntimeError as e:
                raise RuntimeError(f"Failed to load state_proj weights. "
                                f"Check that 'state_proj_config.json' matches 'state_proj.pt'. "
                                f"Detail: {e}")

            model.state_proj = model.state_proj.to(dtype)

        # ---- 5) 返回 ----
        return model.eval()

    def load_llm(self, weight_path):
        pass

    def load_response_head(self, weight_path, weight_dir_prefix: str = "", revision: Optional[str] = None):
            sp_cfg_path = _maybe_hf(
                weight_path,
                "state_proj_config.json",
                revision=revision,
                subfolder=weight_dir_prefix or None
            )
            with open(sp_cfg_path, "r", encoding="utf-8") as f:
                sp_cfg = json.load(f)

            sp_w_path = _maybe_hf(
                weight_path,
                "state_proj.pt",
                revision=revision,
                subfolder=weight_dir_prefix or None
            )
            sd = torch.load(sp_w_path, map_location="cpu")

            # 容错：若权重与结构不完全对齐，给出更友好的报错
            try:
                print(f"Loading state_proj from {sp_w_path} with config {sp_cfg_path}")
                missing, unexpected = self.state_proj.load_state_dict(sd, strict=False)

                if missing or unexpected:
                    print(f"[WARN] state_proj load_state_dict: missing={missing}, unexpected={unexpected}")
            except RuntimeError as e:
                raise RuntimeError(f"Failed to load state_proj weights. "
                                f"Check that 'state_proj_config.json' matches 'state_proj.pt'. "
                                f"Detail: {e}")
            dtype = self.llm.thinker.model.embed_tokens.weight.dtype
            self.state_proj = self.state_proj.to(dtype)

    def shift_position_ids(self, shift_size: int, key_cache, freeze_token_cnt: int = 0):
        if 'qwen2_5' in _MODEL_TAG_MAP[self.model_name_or_path]:
            dummy_tensor =  torch.empty(1, dtype=key_cache[0].dtype, device=key_cache[0].device)
            B, H, T, D = key_cache.shape
            position_ids = torch.zeros((3, B, T-freeze_token_cnt), dtype=torch.long, device=key_cache[0].device).fill_(shift_size)
            position_embeddings = self.llm.thinker.model.rotary_emb(dummy_tensor, position_ids)
            cos, sin = position_embeddings
            _, key_cache_shifted_not_frozen = apply_multimodal_rotary_pos_emb(
                dummy_tensor, 
                key_cache[:,:,freeze_token_cnt:,:], 
                cos, sin, 
                self.llm.config.thinker_config.text_config.rope_scaling['mrope_section']
            )
            key_cache[:,:,freeze_token_cnt:,:] = key_cache_shifted_not_frozen
            return key_cache
        else:
            raise NotImplementedError(f"Model {self.model_name_or_path} not supported yet.")

