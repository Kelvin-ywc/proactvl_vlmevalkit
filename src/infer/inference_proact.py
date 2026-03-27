import argparse
import torch
import logging
from typing import Optional, Dict, Any, List, Deque, Tuple
from collections import deque
import time
from src.model.modeling_proact import ProAct_OmniModel, ProActConfig
from src.utils.utils import print_model_param_devices, prune_cache_span, sample_from_logits, whether_finish
from src.utils.proact_process import process_interleave_mm_info
from src.utils.conversations import PROACT_MLLM_SYSTEM_PROMPT
import numpy as np
from torchvision.transforms.functional import to_pil_image
import os
from src.utils.utils import _maybe_hf
from kokoro import KPipeline
import soundfile as sf


logging.getLogger().setLevel(logging.ERROR)
logging.basicConfig(
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_INPUT = 'You are the main commentator. Each user section provides a 1-second video segment and optional comments from other speakers in the form of "(SPEAKER_X): ...". Respond as the main commentator in assistant section, providing your own live commentary.'
SYSTEM_PROMPT = f'<|im_start|>system\n{SYSTEM_INPUT}<|im_end|>\n'
USER_PROMPT = '<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n'
ASSISTANT_PROMPT = '<|im_start|>assistant\n'
MIN_PIXELS = 128*28*28
MAX_PIXELS = 128*28*28
'''
streaming input: chunk
1.  prime_system_prompt() -> kv_cache
2.  process_one_chunk(chunk) -> llm.forward(chunk, kv_cache) -> flag, kv_cache
3.  if flag: 
        generate_from_cache(kv_cache) -> llm.generate(kv_cache) -> return output
    else: 
        return [SILENCE]

usage:
prime_system_prompt:
<|im_start|>system\nYou are a professional sports commentary Please given comment on the given video.
user input:
<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>
user input:
<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|> -> active
assistant output:
<|im_end|>\n<|im_start|>assistant\nby the striker!
user input:
<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|> -> active
assistant output:
<|im_end|>\n<|im_start|>assistant\nWhat a fantastic goal by the striker!
user input:
<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|> -> silence
user input:
<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|> -> active
assistant output:
<|im_end|>\n<|im_start|>assistant\nChecking the replay, it's clear that the goal was well deserved!
'''


class StreamInfer:
    """
    能力：
    - prime_with_system(): 打 system 进 KV，保存为 system_kv（固定不淘汰）
    - process_one_chunk(): 单次 forward（带 KV）完成激活检测 + 把当前块写入 KV；触发则生成
    - generate_from_cache(): 基于当前 KV 生成，并把 assistant 输出纳入滑窗
    - KV 上限控制：system 固定；user/assistant 动态历史按 FIFO 滑动淘汰；淘汰后重放重建 KV
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-Omni-7B",
        ckpt_path: str = None,
        enable_audio_output: bool = False,
        use_audio_in_video: bool = False,
        max_kv_tokens: int = 3072,
        state_threshold: float = 0.5,
        add_special_tokens: bool = False,
        weight_dir_prefix: str = '',
        active_layer_id: int = -1,
    ):
        config = ProActConfig(
            model_name_or_path=model_name_or_path,
            enable_audio_output=enable_audio_output,
            state_threshold=state_threshold,
            add_special_tokens=add_special_tokens,
            active_layer_id=active_layer_id,
        )

        self.model_name_or_path = model_name_or_path
        self.use_audio_in_video = use_audio_in_video
        # load core model
        print(f'Loading model {model_name_or_path} {weight_dir_prefix} ...')
        self.model = ProAct_OmniModel.from_pretrained(config, ckpt_path, weight_dir_prefix=weight_dir_prefix).cuda()
        self.model.eval()


        # self.im_start = im_start
        # self.im_end = im_end
        # self.role_user = role_user
        # self.role_assistant = role_assistant
        # self.vision_bos = vision_bos
        # self.vision_eos = vision_eos
        # self.video_tok = video_tok

        self.tokenizer = self.model.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.new_line_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        # 预算参数
        self.max_kv_tokens = max_kv_tokens # system kv + user kv + assistant kv
        self.evict_percent: float = 0.20    # 一次至少腾出 20% 容量
        # 推理时需要更新的参数
        # 当前有效 KV
        self.kv = None
        # system 基座 KV（不淘汰）
        self.system_token_cost = 0 # system kv length
        # 动态滑窗（user/assistant 历史），每条：(entry_type, text, mm_payload, token_cost)
        self.token_counts:Deque[int] = deque()   # e.g. [40, 141, 20, 141, ...]
        self.dynamic_token_cost = 0 # sum of self.token_counts
        # 记录 past ids，方便 generate 时拼接
        self.past_ids = None
        self.window_position_ids_for_stream = torch.empty((3, 1, 0), dtype=torch.long).cuda()  # for stream infer

        # for debug
        self.vis_video = False
        self.vis_attention_score = False

        # # 是否累计response
        # self.accumulated_response = accumulated_response
        # self.last_response = ''
        # self.last_response_tokens = []
        # 用于计算postion ids, 记录当前最后一个序列最大的postion ids
        self.max_position_id = -1 

        # 保存历史会话
        self.save_history = True
        self.history = []
        # 填充系统提示
        self.prime_with_system(system_prompt=SYSTEM_PROMPT)

    # ---------- 系统 prime, 对话开始填充内容 ----------
    @torch.no_grad()
    def prime_with_system(self, system_prompt_chat_template=None, system_prompt=None):
        if system_prompt_chat_template is not None:
            system_prompt = self.model.processor.apply_chat_template(
                conversations=[system_prompt],
                add_generation_prompt=False,
                tokenize=False,
            )
        
        sys_inputs = self.model.processor(
            text=system_prompt,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.llm.device).to(self.model.llm.dtype)

        position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, **sys_inputs)

        out = self.model.forward(
            **sys_inputs,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=True,
            output_active_logits=False
        )
        # post process, update kv cache, parameters
        self.max_position_id = position_ids.max().item()

        self.kv = getattr(out, "past_key_values", None)
        self.system_token_cost = sys_inputs['input_ids'].shape[-1]

        if self.save_history:
            self.history.append(
                {
                    'role': 'system',
                    'content': system_prompt,
                }
            )

    def process_system_prompt(self, system_prompt):
        pass

    def clear_session(self):
        self.kv = None
        self.system_token_cost = 0
        self.token_counts.clear()
        self.dynamic_token_cost = 0
        self.rope_deltas = None
        self.last_response = ''
        # 内存回收
        torch.cuda.empty_cache()

    def process_one_chunk_frames(self, frames: List[np.ndarray]) -> str:
        # res = f"处理了一秒的视频帧，帧数: {len(frames)}, shape: {frames[0].shape}"
        # List[np.ndarray] -> torch.Tensor
        if self.kv is None:
            print("请先调用 prime_with_system() 打入 system prompt！, 开始调用默认system prompt")
            self.prime_with_system()
        frames = [torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in frames], dim=0).float()]
        res = self.process_one_raw_chunk(audios=None, images=None, videos=frames, max_new_tokens=6)
        return res

    def _ensure_budget(self):
        if self.dynamic_token_cost > self.max_kv_tokens - self.system_token_cost:

            cur_pop_num = 0
            while cur_pop_num < int(self.max_kv_tokens*self.evict_percent):
                cur_pop_num += self.token_counts.popleft()
            print(f'当前kv长度{self.kv.key_cache[0].shape[2]}，超出预算{self.dynamic_token_cost + self.system_token_cost - self.max_kv_tokens}，淘汰{cur_pop_num}个token')

            window_position_ids_for_stream = self.window_position_ids_for_stream
            self.window_position_ids_for_stream = window_position_ids_for_stream[:, :, cur_pop_num:]

            shift_size = self.window_position_ids_for_stream[0][0][0].item() - (self.system_token_cost+1)  # 取第一个token的position id作为shift size
            self.window_position_ids_for_stream -= shift_size
            self.max_position_id = self.window_position_ids_for_stream[0][0][-1].item()

            logger.info(f'更新kv缓存，弹出{cur_pop_num}个token')
            # 更新kv缓存
            self.dynamic_token_cost -= cur_pop_num
            self.kv = prune_cache_span(self.kv, self.system_token_cost, self.system_token_cost + cur_pop_num)

            assert self.kv.key_cache[0].shape[2] == self.dynamic_token_cost + self.system_token_cost, f"kv长度不匹配，期望{self.dynamic_token_cost + self.system_token_cost}，实际{self.kv.key_cache[0].shape[-1]}"
            assert self.dynamic_token_cost == self.window_position_ids_for_stream.shape[-1], f'window position ids长度不匹配，期望{self.dynamic_token_cost}，实际{self.window_position_ids_for_stream.shape[-1]}'
            for i in range(len(self.kv.key_cache)):
                self.kv.key_cache[i] = self.model.shift_position_ids(-shift_size, self.kv.key_cache[i], self.system_token_cost)

            assert self.kv.key_cache[0].shape[2] <= self.dynamic_token_cost + self.system_token_cost, \
                f"kv长度不匹配，期望{self.dynamic_token_cost + self.system_token_cost}，实际{self.kv.key_cache[0].shape[2]}"


    @torch.no_grad()
    def process_one_raw_chunk(
        self, 
        audios, images, videos, 
        generate_config: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> str:
        answer = "<SILENCE>"

        time1 = time.time()
        # print(f'video length: {len(videos)}')
        # print(f'video shape: {videos[0].shape}')
        if self.vis_video:
            save_dir = './tmp/infer_output'
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(videos[0]):
                pil_img = to_pil_image(frame)
                pil_img.save(os.path.join(save_dir, f'frame_{prefix}_{i}.png'))
            self.vis_video = False

        det_inputs = self.model.processor(
            text=USER_PROMPT,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        ).to(self.model.llm.device).to(self.model.llm.dtype)
 
        # cache_position = torch.arange(self.kv.get_seq_length(), self.kv.get_seq_length() + det_inputs['input_ids'].shape[1], device=self.model.llm.device)
        position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, **det_inputs)

        self.max_position_id = position_ids[0][0][-1].item()

        det_out = self.model(
            **det_inputs,
            position_ids=position_ids,
            output_hidden_states=False,
            output_attentions=self.vis_attention_score,
            return_dict=True,
            use_cache=True,
            past_key_values=self.kv,
            output_active_logits=True
            # cache_position=cache_position,
        )

        if self.vis_attention_score:
            # 可视化 attention score以及对应的token
            attn_weights = det_out.cross_attentions[-1] # 取最后一层 cross attention
            print(f'attn_weights shape: {attn_weights.shape}') # (batch_size, num_heads, seq_len, kv_seq_len)
            exit(0)

        self.window_position_ids_for_stream = torch.cat([self.window_position_ids_for_stream, position_ids], dim=-1)
        # self.window_position_ids_for_stream: [3, bs, T], self.kv.key_cache[0]: [bs, num_heads, T, head_dim]
        # print(self.window_position_ids_for_stream.shape[2], self.kv.key_cache[0].shape[2], self.system_token_cost)
        assert self.window_position_ids_for_stream.shape[2] + self.system_token_cost == self.kv.key_cache[0].shape[2]
        assert self.tokenizer.decode(det_inputs['input_ids'][0][-2]) == self.model.chunk_flag, f"输入的最后一个token应为{self.model.chunk_flag}，实际为{self.tokenizer.decode(det_inputs['input_ids'][0][-1])}"
        active_token_in_one_chunk = torch.sigmoid(det_out.active_logits[0])[-2]
        # print(f'active_token_in_one_chunk: {active_token_in_one_chunk}')
        flag = (active_token_in_one_chunk > self.model.state_threshold)
        # flag = False # FIXME for debug
        

        self.kv = getattr(det_out, "past_key_values", None)
        print(f'当前kv长度: {self.kv.key_cache[0].shape[2]}')
        cur_token_num = det_inputs['input_ids'].shape[-1]
        self.token_counts.append(cur_token_num)
        self.dynamic_token_cost += cur_token_num

        if self.save_history:
            self.history.append(
                {
                    'role': 'user',
                    'content': USER_PROMPT,
                }
            )
        # print(f'token counts: {self.token_counts}, dymanic token cost: {self.dynamic_token_cost}, kv cache shape: {self.kv.key_cache[0].shape if self.kv else None}')
        # 仅在每次读多模态chunk的时候更新kv缓存长度
        if True:
            self._ensure_budget()
        time2 = time.time()
        print(f'激活判断时间: {time2 - time1:.2f}s')

        # is_finish = whether_finish(self.last_response)
        # if is_finish:
        #     self.last_response = ''

        if not flag:
            print(f"{prefix} SILENCE TOKEN ({active_token_in_one_chunk.item()}) (MAX POSITION ID: {self.max_position_id}) -> 已把当前块编码进KV，继续读取下一块")
            # self.last_response = ''
        else:
            answer = self.generate_from_cache(
                generate_config=generate_config,
            )
            print(f"{prefix} ACTIVE TOKEN ({active_token_in_one_chunk.item()}) (MAX POSITION ID: {self.max_position_id}) -> {answer}")
            time1 = time.time()
            print(f'生成时间: {time1 - time2:.2f}s')
        # print(self.window_position_ids_for_stream[0][0].tolist())
        return answer
      
    # ---------- 单次块处理 ----------
    @torch.no_grad()
    def process_one_chunk(
        self,
        chunk_info,
        max_new_tokens: int = 120,
        temperature: float = 0.95,
        top_p: float = 0.95,
    ):
        begin_sec = chunk_info[0]['content'][0].get('video_start', None)
        end_sec = chunk_info[0]['content'][0].get('video_end', None)
        prefix = f"[Sec: {begin_sec} to {begin_sec+1}] " if begin_sec is not None and end_sec is not None else ""

        time1 = time.time()
        audios, images, videos = process_interleave_mm_info(chunk_info, self.use_audio_in_video, return_video_kwargs=False)
        time2 = time.time()
        # print(f'视频读取时间: {time2 - time1:.2f}s')
        answer = self.process_one_raw_chunk(
            audios=audios, 
            images=images, 
            videos=[videos[0]], # FIXME qwen render读取一秒视频会处理成4fps，这里读两秒取第一秒
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            prefix=prefix
        )
        return answer

    # ---------- 生成 ----------
    @torch.no_grad()
    def generate_from_cache(self, generate_config=None) -> str:
        # 1. prime assistant prompt，调用model.forward()，更新 self.kv, 注意这里assistant prompt只取到倒数第二个token，最后一个token最为generate函数的输入。
        # input_prompt = ASSISTANT_PROMPT + self.last_response if self.accumulated_response else ASSISTANT_PROMPT
        input_prompt = ASSISTANT_PROMPT
        inputs = self.model.processor(
            text=input_prompt, return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.llm.device).to(self.model.llm.dtype)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids_for_cache = input_ids[:, :-1]  # 不含最后一个 token
        input_ids_for_generate = input_ids[:, -1:]  # 仅最后一个 token，作为 generate 的输入
        attention_mask_for_cache = attention_mask[:, :-1]
        attention_mask_for_generate = attention_mask[:, -1:]

        init_max_position_id = self.max_position_id
        position_ids_for_cache = self.model.get_position_ids(max_position_id=self.max_position_id, input_ids=input_ids_for_cache, attention_mask=attention_mask_for_cache)
        # print(f'position_ids_for_cache: {position_ids_for_cache}')
        # 生成时的position ids取cache的最后一个token位置，当模型调用prepare_inputs_for_generation方法时会先加1.
        position_ids_for_generate = position_ids_for_cache[:, :, -1:].add_(-1)  # 仅最后一个 token
        # print(f'position_ids_for_generate: {position_ids_for_generate}')
        # self.max_position_id = position_ids_for_cache[0][0][-1].item()

        end_token_str = self.tokenizer.eos_token if self.model.processor.tokenizer.eos_token is not None else '<|im_end|>'
        end_id = self.tokenizer.convert_tokens_to_ids(end_token_str)

        device = input_ids.device

        # 2. Prefill：一次性并行，把本次 input_ids 写入缓存 ----------
        prefill_out = self.model(
            input_ids=input_ids_for_cache,
            attention_mask=attention_mask_for_cache,
            position_ids=position_ids_for_cache,
            past_key_values=self.kv,
            use_cache=True,
            return_dict=True,
            output_active_logits=False
        )
        # 有些缓存实现是 in-place 的，这里显式赋值以确保一致
        self.kv = prefill_out.past_key_values

        # 3. Decode：循环采样生成新 token ----------
        generate_config = {
            'do_sample': True,
            **(generate_config if generate_config is not None else {}),
            'eos_token_id': self.model.processor.tokenizer.eos_token_id,
            'pad_token_id': self.model.processor.tokenizer.pad_token_id,
        }

        gen_output = self.model.generate(
            input_ids=input_ids_for_generate,                   # 仅最近一个 token
            attention_mask=attention_mask_for_generate,         # 仅最近一个 token
            position_ids=position_ids_for_generate,
            cache_position=torch.tensor([[self.kv.get_seq_length()]], device=device),
            past_key_values=self.kv,
            use_cache=True,
            output_active_logits=False, # 新增
            **generate_config
        )
        # assert self.tokenizer.decode(gen_output[0][-1]) == self.tokenizer.eos_token, f"生成的最后一个token应为{self.tokenizer.eos_token}，实际为{self.tokenizer.decode(gen_output[0][-1])}"
        if not self.tokenizer.decode(gen_output[0][-1]) == self.tokenizer.eos_token:
            print(f'主动截停生成，未出eos，已达max_new_tokens')
        new_add_token_num = (input_ids.shape[-1] - 1) + (gen_output.shape[-1] - 1)  # assistant prompt 除去最后一个token + 生成的token除去最后一个token
        new_add_token = torch.cat([input_ids_for_cache, gen_output[:, :-1]], dim=-1)  # 拼接成“本轮新增但不含最后一个token”的ids
        history_content = self.tokenizer.decode(new_add_token[0])
        response = self.tokenizer.decode(gen_output[0][1:-1])  # 除去最后一个token
        self.dynamic_token_cost += new_add_token_num
        self.token_counts.append(new_add_token_num)

        # print(self.window_position_ids_for_stream)
        # print(torch.arange(init_max_position_id + 1, self.max_position_id + new_add_token_num + 1, device=device))
        new_append_position_ids = torch.arange(init_max_position_id + 1, self.max_position_id + new_add_token_num + 1, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        self.window_position_ids_for_stream = torch.cat([self.window_position_ids_for_stream, new_append_position_ids], dim=-1)

        self.max_position_id = self.max_position_id + new_add_token_num
        # print(self.window_position_ids_for_stream.shape[2], self.kv.key_cache[0].shape[2], self.system_token_cost)
        assert self.window_position_ids_for_stream.shape[2] + self.system_token_cost == self.kv.key_cache[0].shape[2]

        # 补上结尾<im_end|>\n
        end_prompt = '<|im_end|>\n'
        end_inputs = self.model.processor(
            text=end_prompt, return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.llm.device).to(self.model.llm.dtype)
        end_max_position_id = self.max_position_id
        end_position_ids = self.model.get_position_ids(max_position_id=self.max_position_id, input_ids=end_inputs['input_ids'], attention_mask=end_inputs['attention_mask'])

        end_out = self.model(
            input_ids=end_inputs['input_ids'],
            attention_mask=end_inputs['attention_mask'],
            position_ids=end_position_ids,
            past_key_values=self.kv,
            use_cache=True,
            return_dict=True,
            output_active_logits=False
        )

        self.kv = end_out.past_key_values
        # for window shift
        self.dynamic_token_cost += end_inputs['input_ids'].shape[-1]
        self.token_counts.append(end_inputs['input_ids'].shape[-1])
        # for position ids
        self.max_position_id = end_position_ids[0][0][-1].item()
        end_append_position_ids = torch.arange(end_max_position_id + 1, self.max_position_id + 1, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        # print(f'end_position_ids: {end_position_ids}, end_append_position_ids: {end_append_position_ids}')
        self.window_position_ids_for_stream = torch.cat([self.window_position_ids_for_stream, end_append_position_ids], dim=-1)
        assert self.window_position_ids_for_stream.shape[2] + self.system_token_cost == self.kv.key_cache[0].shape[2]

        if self.save_history:
            self.history.append(
                {
                    'role': 'assistant',
                    'content': history_content + '<|im_end|>\n',
                }
            )
        
        return response


def merge_tts_segments(output_dir: str, pattern_prefix: str = 'tts_', outfile: str = 'tts_all.wav', samplerate: int = 24000, crossfade_ms: int = 0):
    try:
        if not os.path.isdir(output_dir):
            return
        files = [f for f in os.listdir(output_dir) if f.startswith(pattern_prefix) and f.endswith('.wav')]
        def _idx(name: str):
            core = name[len(pattern_prefix):-4]
            try:
                return int(core)
            except Exception:
                return float('inf')
        files = sorted(files, key=_idx)
        segments: List[np.ndarray] = []
        for f in files:
            path = os.path.join(output_dir, f)
            data, sr = sf.read(path, dtype='float32')
            if sr != samplerate:
                print(f'skip {f}, samplerate {sr} != {samplerate}')
                continue
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            segments.append(data)
        if len(segments) == 0:
            return
        if crossfade_ms and crossfade_ms > 0:
            fade = int(samplerate * (crossfade_ms / 1000.0))
        else:
            fade = 0
        merged = segments[0]
        for seg in segments[1:]:
            if fade > 0 and len(merged) > fade and len(seg) > fade:
                fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                tail = merged[-fade:] * fade_out
                head = seg[:fade] * fade_in
                overlapped = tail + head
                merged = np.concatenate([merged[:-fade], overlapped, seg[fade:]])
            else:
                merged = np.concatenate([merged, seg])
        sf.write(os.path.join(output_dir, outfile), merged, samplerate)
        print(f"已输出合并音频: {os.path.join(output_dir, outfile)}")
    except Exception as e:
        print(f"合并 TTS 音频失败：{e}")


# ----------------- 推理主循环 -----------------
def run_inference_loop_stream(
    stream_infer,
    args: argparse.Namespace = None,
    duration: int = 120,
):

    # ---- TTS 初始化 ----
    # 参数说明：
    # enable_tts: 是否启用 TTS（True/False）
    # tts_voice: Kokoro 声音 ID（例如：af_heart）
    # tts_words_per_sec: 每秒吞入词/字数（控制累计速度）
    # tts_min_words: 触发合成的最小词数（增强完整性）
    # tts_max_words: 强制触发的最大词数（防止无限累计）
    # tts_wait_sec: 两次触发之间的最小秒数（控制节奏）
    # tts_crossfade_ms: 合并 tts_*.wav 为 tts_all.wav 的交叉淡入时长（毫秒）
    enable_tts = getattr(args, 'enable_tts', False)
    tts_voice = getattr(args, 'tts_voice', 'af_heart')
    tts_words_per_sec = max(0, int(getattr(args, 'tts_words_per_sec', 5)))
    tts_min_words = max(1, int(getattr(args, 'tts_min_words', 8)))
    tts_max_words = max(tts_min_words, int(getattr(args, 'tts_max_words', 30)))
    tts_wait_sec = max(1, int(getattr(args, 'tts_wait_sec', 2)))
    tts_crossfade_ms = int(getattr(args, 'tts_crossfade_ms', 0))
    tts_pipeline = None
    word_buffer: List[str] = []
    phrase_buffer: List[str] = []
    last_speak_i = -1
    punctuation_chars = set("。！？!?;；…,.，、:：")
    if enable_tts and tts_words_per_sec > 0:
        try:
            tts_pipeline = KPipeline(lang_code='a')
        except Exception as e:
            print(f"初始化 Kokoro 失败：{e}. 将跳过 TTS。")
            enable_tts = False
    os.makedirs(args.output_path, exist_ok=True)

    def _split_words(text: str) -> List[str]:
        if text is None:
            return []
        cleaned = text.replace('\n', ' ').replace('<|im_end|>', ' ').strip()
        parts = [w for w in cleaned.split() if len(w) > 0]
        if len(parts) == 1 and any('\u4e00' <= ch <= '\u9fff' for ch in parts[0]):
            return [ch for ch in parts[0] if ch.strip()]
        return parts

    def _ends_with_punct(words: List[str]) -> bool:
        if not words:
            return False
        last = words[-1]
        return len(last) > 0 and last[-1] in punctuation_chars

    for i in range(duration):
        chunk_info = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": args.input_video_path,
                        "video_start": i,
                        "video_end": i + 2
                    },
                ]
            }
        ]
        ans = stream_infer.process_one_chunk(chunk_info, max_new_tokens=12, temperature=0.9, top_p=0.9)
        # ---- 更连贯聚合 ----
        if enable_tts and ans and ans != '<SILENCE>':
            word_buffer.extend(_split_words(ans))
        if enable_tts and tts_pipeline is not None and tts_words_per_sec > 0:
            take_n = min(tts_words_per_sec, len(word_buffer))
            if take_n > 0:
                phrase_buffer.extend(word_buffer[:take_n])
                del word_buffer[:take_n]
            should_speak = False
            if len(phrase_buffer) >= tts_max_words:
                should_speak = True
            elif len(phrase_buffer) >= tts_min_words and (i - last_speak_i) >= tts_wait_sec:
                should_speak = True
            elif _ends_with_punct(phrase_buffer):
                should_speak = True
            if should_speak and len(phrase_buffer) > 0:
                say_text = ' '.join(phrase_buffer)
                try:
                    gen = tts_pipeline(say_text, voice=tts_voice)
                    audio_chunks = []
                    for _, _, audio in gen:
                        audio_chunks.append(audio)
                    if len(audio_chunks) > 0:
                        audio = np.concatenate(audio_chunks)
                        sf.write(os.path.join(args.output_path, f'tts_{i}.wav'), audio, 24000)
                except Exception as e:
                    print(f"Kokoro 合成失败（第 {i} 秒）：{e}")
                phrase_buffer.clear()
                last_speak_i = i
    # 合并所有秒级音频为单一文件（若存在）
    # 收尾
    if enable_tts and tts_pipeline is not None and len(phrase_buffer) > 0:
        say_text = ' '.join(phrase_buffer)
        try:
            gen = tts_pipeline(say_text, voice=tts_voice)
            audio_chunks = []
            for _, _, audio in gen:
                audio_chunks.append(audio)
            if len(audio_chunks) > 0:
                audio = np.concatenate(audio_chunks)
                sf.write(os.path.join(args.output_path, f'tts_{duration}.wav'), audio, 24000)
        except Exception as e:
            print(f"Kokoro 合成失败（收尾）：{e}")
    merge_tts_segments(args.output_path, crossfade_ms=tts_crossfade_ms)
    print(f'history: {stream_infer.history}')

def run_inference_loop(
    stream_infer,
    args: argparse.Namespace = None,
    duration: int = 120,
    generate_config: Optional[Dict[str, Any]] = None,
    prefix: str = "",
):
    if args.input_video_path is None:
        # FIXME read from huggingface dataset, hard coding
        video_default_path = "asset/2025MSI_T1_vs_GEN_game5.mp4"
        print(f'No input video path provided, using default video from hf dataset: {os.path.join(args.ckpt_path, video_default_path)}.')
        args.input_video_path = _maybe_hf(args.ckpt_path, video_default_path, revision=None)
    print(f'input video path: {args.input_video_path}')
    video_info = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.input_video_path,
                    "video_start": 0,
                    "video_end": duration,
                    "nframes": duration * 2,
                }
            ]
        }
    ]
    audios, images, videos = process_interleave_mm_info(video_info, stream_infer.use_audio_in_video, return_video_kwargs=False)
    # ---- TTS 初始化 ----
    enable_tts = getattr(args, 'enable_tts', False)
    tts_voice = getattr(args, 'tts_voice', 'af_heart')
    tts_words_per_sec = max(0, int(getattr(args, 'tts_words_per_sec', 5)))
    tts_min_words = max(1, int(getattr(args, 'tts_min_words', 8)))
    tts_max_words = max(tts_min_words, int(getattr(args, 'tts_max_words', 30)))
    tts_wait_sec = max(1, int(getattr(args, 'tts_wait_sec', 2)))
    tts_pipeline = None
    word_buffer: List[str] = []
    phrase_buffer: List[str] = []
    last_speak_i = -1
    punctuation_chars = set("。！？!?;；…,.，、:：")
    if enable_tts and tts_words_per_sec > 0:
        try:
            tts_pipeline = KPipeline(lang_code='a')
        except Exception as e:
            print(f"初始化 Kokoro 失败：{e}. 将跳过 TTS。")
            enable_tts = False
    os.makedirs(args.output_path, exist_ok=True)

    def _split_words(text: str) -> List[str]:
        if text is None:
            return []
        # 粗粒度按空白切词，去除特殊标记
        cleaned = text.replace('\n', ' ').replace('<|im_end|>', ' ').strip()
        parts = [w for w in cleaned.split() if len(w) > 0]
        # 中文无空格的回退切分
        if len(parts) == 1 and any('\u4e00' <= ch <= '\u9fff' for ch in parts[0]):
            return [ch for ch in parts[0] if ch.strip()]
        return parts

    def _ends_with_punct(words: List[str]) -> bool:
        if not words:
            return False
        last = words[-1]
        return len(last) > 0 and last[-1] in punctuation_chars
    for i in range(duration):
        one_chunk_audios = audios[i:i+1] if audios is not None else None
        one_chunk_images = images[i:i+1] if images is not None else None
        one_chunk_videos = videos[i:i+1] if videos is not None else None
        prefix = f"[Sec: {i} to {i+1}] "
        answer = stream_infer.process_one_raw_chunk(
            audios=one_chunk_audios, 
            images=one_chunk_images, 
            videos=one_chunk_videos, 
            generate_config=generate_config,
            prefix=prefix
        )
        # ---- 累计生成的词并按短语触发播报（更连贯） ----
        if enable_tts and answer and answer != '<SILENCE>':
            word_buffer.extend(_split_words(answer))
        if enable_tts and tts_pipeline is not None and tts_words_per_sec > 0:
            take_n = min(tts_words_per_sec, len(word_buffer))
            if take_n > 0:
                phrase_buffer.extend(word_buffer[:take_n])
                del word_buffer[:take_n]
            should_speak = False
            if len(phrase_buffer) >= tts_max_words:
                should_speak = True
            elif len(phrase_buffer) >= tts_min_words and (i - last_speak_i) >= tts_wait_sec:
                should_speak = True
            elif _ends_with_punct(phrase_buffer):
                should_speak = True
            if should_speak and len(phrase_buffer) > 0:
                say_text = ' '.join(phrase_buffer)
                try:
                    gen = tts_pipeline(say_text, voice=tts_voice)
                    audio_chunks = []
                    for _, _, audio in gen:
                        audio_chunks.append(audio)
                    if len(audio_chunks) > 0:
                        audio = np.concatenate(audio_chunks)
                        sf.write(os.path.join(args.output_path, f'tts_{i}.wav'), audio, 24000)
                except Exception as e:
                    print(f"Kokoro 合成失败（第 {i} 秒）：{e}")
                phrase_buffer.clear()
                last_speak_i = i
        # if (i+1) % 30 == 0:
        #     stream_infer.prime_with_system(system_prompt=SYSTEM_PROMPT)
    # 循环结束如仍有残留短语，补一次合成
    if enable_tts and tts_pipeline is not None and len(phrase_buffer) > 0:
        say_text = ' '.join(phrase_buffer)
        try:
            gen = tts_pipeline(say_text, voice=tts_voice)
            audio_chunks = []
            for _, _, audio in gen:
                audio_chunks.append(audio)
            if len(audio_chunks) > 0:
                audio = np.concatenate(audio_chunks)
                # 使用 i+1 命名避免覆盖
                sf.write(os.path.join(args.output_path, f'tts_{duration}.wav'), audio, 24000)
        except Exception as e:
            print(f"Kokoro 合成失败（收尾）：{e}")
    # 合并所有秒级音频为单一文件（若存在）
    merge_tts_segments(args.output_path, crossfade_ms=getattr(args, 'tts_crossfade_ms', 0))
    print(stream_infer.history[-10:])
    print("Inference loop finished.")
    return answer

def infer(args):
    stream_infer = StreamInfer(
        args.model_name_or_path, 
        args.ckpt_path, 
        args.enable_audio_output, 
        args.use_audio_in_video, 
        state_threshold=args.state_threshold,
        weight_dir_prefix=args.weight_dir_prefix,
        active_layer_id=args.active_layer_id,
    )
    generate_config = {
        'max_new_tokens': 12,
        'temperature': 0.9,
        'top_p': 0.9,
        'repetition_penalty': 1.15,
    }
    run_inference_loop(stream_infer, args=args, duration=120, generate_config=generate_config)
    # stream_infer.clear_session()
    # run_inference_loop(stream_infer, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Omni-7B')
    parser.add_argument('--enable_audio_output', action='store_true', help='Enable audio output')
    parser.add_argument('--ckpt_path', type=str, default='/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/OUTPUT/exp_20251018-173212/final')
    parser.add_argument('--output_path', type=str, default='./infer_output')
    # /home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/SoccerNet/videos/SoccerNet/spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid/2_224p.mkv
    parser.add_argument('--input_video_path', type=str, default=None, help='Path to input video file')
    parser.add_argument('--use_audio_in_video', action='store_true', help='Use audio in video')
    parser.add_argument('--state_threshold', type=float, default=0.5, help='Threshold for state prediction to trigger response generation')
    parser.add_argument('--weight_dir_prefix', type=str, default='', help='Weight directory prefix')
    parser.add_argument('--active_layer_id', type=int, default=-1, help='Layer id to extract active logits, -1 means the last layer')
    # parser.add_argument('--accumulated_response', type=bool, default=False, help='Whether to accumulate the response history')
    # TTS 相关参数
    parser.add_argument('--enable_tts', action='store_true', help='Enable Kokoro TTS for streaming output')
    parser.add_argument('--tts_voice', type=str, default='af_heart', help='Kokoro voice id, e.g., af_heart')
    parser.add_argument('--tts_words_per_sec', type=int, default=5, help='Number of words to speak per second')
    parser.add_argument('--tts_min_words', type=int, default=8, help='Minimum buffered words to trigger TTS')
    parser.add_argument('--tts_max_words', type=int, default=30, help='Maximum buffered words before forced TTS')
    parser.add_argument('--tts_wait_sec', type=int, default=2, help='Minimum seconds between TTS triggers')
    parser.add_argument('--tts_crossfade_ms', type=int, default=0, help='Crossfade duration when merging wavs')
    args = parser.parse_args()
    if False:
        from src.utils.utils import debug
        debug(9501)
    infer(args)
