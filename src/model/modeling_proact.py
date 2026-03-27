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
# from src.utils.utils import _maybe_hf, _is_local_dir
# from huggingface_hub import hf_hub_download

from transformers import AutoProcessor
from src.model.registry import build_model

from src.model import WrapQwen3VL, WrapQwen2_5VL, WrapQwen2VL
from peft import LoraConfig, TaskType, get_peft_model
# from src.model import WrapQwen3VLProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

_MODEL_TAG_MAP = {
    "Qwen/Qwen2.5-Omni-7B": 'qwen2_5omni',
    "Qwen/Qwen3-VL-8B-Instruct": 'qwen3vl',
    "Qwen/Qwen3-VL-2B-Instruct": 'qwen3vl',
    "Qwen/Qwen2.5-VL-7B-Instruct": 'qwen2_5vl',
    "Qwen/Qwen2-VL-7B-Instruct": 'qwen2vl',
    "chenjoya/LiveCC-7B-Instruct": "qwen2vl",
    "chenjoya/LiveCC-7B-Base": "qwen2vl",
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
    active_layer_id: int = -2
    torch_dtype = torch.bfloat16
    attn_implementation: str = 'flash_attention_2'
    low_cpu_mem_usage: bool = True
    # deprecated 
    #move to training args
    loss_active_scale: float = 1.0
    finetune_strategy: str = 'none'
    # move to infer config
    state_threshold: float = 0.5




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
            # self.processor = AutoProcessor.from_pretrained(
            #     self.model_name_or_path
            # )
            # self.state_proj = Qwen2_5OmniMLP(self.llm.thinker.config.text_config.hidden_size, 1)
            # # self.llm.state_proj = self.state_proj
            # self.active_eos_token = ''
            # self.silence_eos_token = '<|SILENCE|>'

        elif self.model_name_or_path in ['Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct']:
            self.llm = WrapQwen3VL.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
            )
        elif self.model_name_or_path in ['Qwen/Qwen2.5-VL-7B-Instruct', 'mit-han-lab/StreamingVLM']:
            self.llm = WrapQwen2_5VL.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
            )
        elif self.model_name_or_path in ['chenjoya/LiveCC-7B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct', 'chenjoya/LiveCC-7B-Base']:
            self.llm = WrapQwen2VL.from_pretrained(
                self.model_name_or_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
            )
        self.active_eos_token = ' ...'
        self.silence_eos_token = ' ...'

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.processor.tokenizer.add_tokens(['<|elongated|>', '<|short_break|>', '<|long_break|>', '<|laugh|>'])
        self.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": ['<|query_start|>', '<|query_end|>', '<|history_start|>', '<|history_end|>', '<|FLAG|>']
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
        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = False
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            except TypeError:
                self.llm.gradient_checkpointing_enable()
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

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

    @staticmethod
    def _unfreeze_parameters(module: nn.Module):
        """
        Unfreeze all parameters in the given module.
        """
        for name, param in module.named_parameters():
            param.requires_grad = True

    def set_threshold(self, threshold):
        if hasattr(self, 'state_threhold'):
            self.state_threhold = threshold
            logger.info(f'Set state_threhold to {self.state_threhold}')
        else:
            logger.warning('No state_threhold attribute to set.')

    '''
    训练时，forward 计算 main loss + active loss, main loss计算预测文本的精度，active loss计算预测active label的精度
    可能存在一种情况，即active label全为0（即全为silent），此时不计算main loss
    推理时，
    '''
    def forward(self, active_labels=None, output_active_logits=True, *args, **kwargs):
        if output_active_logits:
            # 如果要输出active_logits,需要拿到hidden_states进行计算，将其设置为True
            kwargs['output_hidden_states'] = True

        output = self.llm(*args, **kwargs)

        if output_active_logits and output.hidden_states is not None:
            
            layer_id = self.active_layer_id
            last_hidden_state = output.hidden_states[layer_id]
            if 'qwen2_5omni' in _MODEL_TAG_MAP[self.model_name_or_path]:
                active_logits = self.state_proj(last_hidden_state)
            else:
                active_logits = self.llm.state_proj(last_hidden_state)
            output.active_logits = active_logits
            output['active_logits'] = active_logits

        if 'output_hidden_states' in kwargs and kwargs['output_hidden_states'] is False:
            output.hidden_states = None
        torch.cuda.empty_cache()
        
        return output

    def get_position_ids(self, **kwargs):
        return self.llm.get_position_ids(**kwargs)

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

    def shift_position_ids(self, shift_size: int, key_cache, freeze_token_cnt: int = 0):
        return self.llm.shift_position_ids(shift_size, key_cache, freeze_token_cnt)
        # if 'qwen2_5' in _MODEL_TAG_MAP[self.model_name_or_path]:
        #     dummy_tensor =  torch.empty(1, dtype=key_cache[0].dtype, device=key_cache[0].device)
        #     B, H, T, D = key_cache.shape
        #     position_ids = torch.zeros((3, B, T-freeze_token_cnt), dtype=torch.long, device=key_cache[0].device).fill_(shift_size)
        #     position_embeddings = self.llm.thinker.model.rotary_emb(dummy_tensor, position_ids)
        #     cos, sin = position_embeddings
        #     _, key_cache_shifted_not_frozen = apply_multimodal_rotary_pos_emb(
        #         dummy_tensor, 
        #         key_cache[:,:,freeze_token_cnt:,:], 
        #         cos, sin, 
        #         self.llm.config.thinker_config.text_config.rope_scaling['mrope_section']
        #     )
        #     key_cache[:,:,freeze_token_cnt:,:] = key_cache_shifted_not_frozen
        #     return key_cache
        # else:
        #     return self.llm.shift_position_ids(shift_size, key_cache, freeze_token_cnt)
            # raise NotImplementedError(f"Model {self.model_name_or_path} not supported yet.")

    def save_pretrained(self, output_dir: str, **kwargs):
        """
        简化版：
        - 保存 ProActConfig 到 output_dir/config.json
        - 如果 self.llm 是 PeftModel：只保存 LoRA adapter 到 output_dir/llm_adapter
        - 否则：保存完整 llm 到 output_dir/llm
        - processor 总是保存到 output_dir/processor
        """
        output_dir = os.fspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 1) 保存 config（外壳结构信息）
        if hasattr(self, "config") and isinstance(self.config, PretrainedConfig):
            self.config.save_pretrained(output_dir)

        # 2) 保存 llm
        if isinstance(self.llm, PeftModel):
            adapter_dir = os.path.join(output_dir, "llm_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            logger.info(f"[ProAct] Saving LoRA adapter to {adapter_dir}")
            self.llm.save_pretrained(adapter_dir)  # 这里会只存 adapter + adapter_config
        else:
            llm_dir = os.path.join(output_dir, "llm")
            os.makedirs(llm_dir, exist_ok=True)
            logger.info(f"[ProAct] Saving full llm to {llm_dir}")
            self.llm.save_pretrained(llm_dir, max_shard_size="3600MB",)

        # 3) 保存 processor
        proc_dir = os.path.join(output_dir, "processor")
        os.makedirs(proc_dir, exist_ok=True)
        logger.info(f"[ProAct] Saving processor to {proc_dir}")
        self.processor.save_pretrained(proc_dir)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """
        简化版 from_pretrained：
        - 先用 ProActConfig.from_pretrained 恢复 config 和 model_name_or_path
        - __init__ 按 config.model_name_or_path 重新加载 base llm（Qwen / WrapQwen）
        - 如果有 llm_adapter/：用 PeftModel.from_pretrained 在 base llm 上套 LoRA
        - 如果有 llm/：直接把 llm/ 当成完整模型目录重新加载
        - processor 从 processor/ 目录恢复；否则 fallback 到 model_name_or_path
        """
        load_dir = os.fspath(pretrained_model_name_or_path)

        # 1) 加载 config
        config: ProActConfig = kwargs.pop("config", None)
        if config is None:
            config = ProActConfig.from_pretrained(load_dir)

        # 2) 先构造一个“空壳”模型（里面会按 config.model_name_or_path 初始化 self.llm）
        model = cls(config, *model_args, **kwargs)

        # 3) 处理 llm 部分
        adapter_dir = os.path.join(load_dir, "llm_adapter")
        full_llm_dir = os.path.join(load_dir, "llm")

        if os.path.isdir(adapter_dir):
            # 保存的是 LoRA adapter + adapter_config
            logger.info(f"[ProAct] Loading base llm from {config.model_name_or_path} and LoRA adapter from {adapter_dir}")
            base_llm = model.llm  # __init__ 已经按 model_name_or_path 加载好
            model.llm = PeftModel.from_pretrained(base_llm, adapter_dir)
            model.llm = model.llm.merge_and_unload()  # 合并 LoRA 权重，释放内存
        elif os.path.isdir(full_llm_dir):
            # 保存的是完整 llm（比如没用 LoRA）
            logger.info(f"[ProAct] Loading full llm from {full_llm_dir}")
            base_cls = type(model.llm)
            model.llm = base_cls.from_pretrained(
                full_llm_dir, 
                torch_dtype=config.torch_dtype, 
                attn_implementation=config.attn_implementation, 
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
        else:
            logger.warning(
                f"[ProAct] No llm_adapter/ or llm/ found in {load_dir}, "
                f"keep llm as freshly initialized from {config.model_name_or_path}."
            )

        # 4) 处理 processor
        proc_dir = os.path.join(load_dir, "processor")
        if os.path.isdir(proc_dir):
            logger.info(f"[ProAct] Loading processor from {proc_dir}")
            model.processor = AutoProcessor.from_pretrained(proc_dir)
        else:
            logger.info(f"[ProAct] No processor/ found, loading from base {config.model_name_or_path}")
            model.processor = AutoProcessor.from_pretrained(config.model_name_or_path)

        return model