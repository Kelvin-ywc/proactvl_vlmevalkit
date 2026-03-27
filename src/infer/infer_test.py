import os
import json
import numpy as np
from datetime import datetime
import torch

from src.infer.multi_assistant_inference import MultiAssistantStreamInference
from src.utils.utils import _maybe_hf
from src.model.modeling_proact import ProAct_OmniModel, ProActConfig
from src.utils.proact_process import process_interleave_mm_info
import logging

MIN_PIXELS = 128*28*28
MAX_PIXELS = 128*28*28

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def get_video_path_from_local_or_hf(input_path, ckpt_path):
    if input_path == 'default':
        video_default_path = "asset/2025MSI_T1_vs_GEN_game5.mp4"
        print(f'No input video path provided, using default video from hf dataset: {os.path.join(ckpt_path, video_default_path)}.')
        input_video_path = _maybe_hf(ckpt_path, video_default_path, revision=None)
        return input_video_path
    return input_path

# 传入上一轮response，生成这一轮的response
def infer_one_chunk(stream_infer, audios, images, videos, assistant_responses, begin_second, prefix):
    system_prompt = 'Serve as the lone live football commentator. Provide commentary only at notable beats or clear visual changes—decisive plays, transitions, highlights—and stay silent in calm stretches. Aim for realistic, event-focused, context-aware calls.'
    text = f'<|im_start|>system\n{system_prompt}<|im_end|>\n'

    for video in videos:
        text += f'<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n'

    assistant = stream_infer.assistants[0]
    # print(videos)
    videos = torch.load(f'/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/tmp/video.pt')
    videos = [v.to(assistant.model.device).to(torch.bfloat16) for v in videos]
    inputs = assistant.model.processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    ).to(assistant.model.device).to(assistant.model.llm.dtype)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"Batch key: {k}, shape: {v.shape}")
            print(v)
    outputs = assistant.model(
        **inputs,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
        output_active_logits=True
    )
    active_logits = outputs['active_logits'] # [1, 3860, 1]
    active_scores = torch.sigmoid(active_logits)
    chunk_token = '<|im_end|>'
    chunk_token_id = assistant.model.processor.tokenizer.encode(chunk_token, add_special_tokens=False)[0]
    active_scores = active_scores[torch.where(inputs['input_ids']==chunk_token_id)]
    print(active_scores)
    print(f'active_logits: {active_logits}')
    print(f'logits: {outputs["logits"]}')
    return None

def infer(stream_infer, input_video_path, begin_time, duration):
    # first prime all assistants with system prompts
    stream_infer.prime_system_prompts()
    # for a in stream_infer.assistants:
    #     a.prime_custom_
    # process video info to get audios, images, videos
    video_info = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": input_video_path,
                    "video_start": begin_time,
                    "video_end": begin_time + duration,
                    "nframes": duration * 2,
                }
            ]
        }
    ]
    audios, images, videos = process_interleave_mm_info(video_info, stream_infer.use_audio_in_video, return_video_kwargs=False)

    for i in range(duration):
        current_second = begin_time + i
        # one_chunk_audios = audios[i:i+1] if audios is not None else None
        # one_chunk_images = images[i:i+1] if images is not None else None
        # one_chunk_videos = videos[i:i+1] if videos is not None else None
        prefix = f"[Sec: {current_second} to {current_second + 1}] "
        assistant_responses = infer_one_chunk(stream_infer, audios, images, videos, None, current_second, prefix)
        break
    stream_infer.post_audio_generation()
    logger.info("Inference loop finished.")

if __name__ == "__main__":
    if False:
        from src.utils.utils import debug
        debug(9501)
    use_lol = False
    model_task = 'all'
    data_task = 'soccer'
    # configuration
    # ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251027-044755/checkpoint-462'
    # ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251029-071909/final'
    if model_task == 'lol':
        # lol
        # ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251029-131356/final'
        # ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251102-132441/checkpoint-468'
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251103-110409/checkpoint-76'
    elif model_task == 'soccer':
        # soccer
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251030-104132/checkpoint-760'
    elif model_task == 'all':
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251031-021154/checkpoint-932'
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251104-160121/checkpoint-341'
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251104-160121/checkpoint-682'
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251109-061114/checkpoint-341'
        ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251109-061114_strategy1/checkpoint-336'
    weight_dir_prefix = ''
    # model config
    model_config = ProActConfig(
        model_name_or_path = "Qwen/Qwen2.5-Omni-7B",
        enable_audio_output=False,
        state_threshold=0.5,
        add_special_tokens=False,
        active_layer_id=-4,
    )
    # infer config
    infer_config = {
        'use_audio_in_video': False,
        'max_kv_tokens': 4096,
        'assistant_num': 1,
        'enable_tts': False,
        'save_dir': './infer_output',
    }
    generate_config = {
        'do_sample': True,
        'max_new_tokens': 12,
        'temperature': 0.9,
        'top_p': 0.9,
        'repetition_penalty': 1.15,
    }
    talker_config = {
        'tts_voice': 'af_heart', # ['af_heart', 'af_van', 'cf_moon', 'cf_sun', 'df_alice', 'df_bob']
        'tts_words_per_sec': 5, # Number of words to speak per second
        'tts_min_words': 8, # Minimum buffered words to trigger TTS
        'tts_max_words': 30, # Maximum buffered words before forced TTS
        'tts_wait_sec': 2, # Minimum seconds between TTS triggers
        'tts_crossfade_ms': 0, # Crossfade duration when merging wavs
        # 'save_dir': './infer_output',
    }
    if data_task == 'lol':
        # lol
        input_video_path = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_T1_vs_GEN/2025MSI_T1_vs_GEN_game5.mp4'
    elif data_task == 'soccer':
        # soccer
        input_video_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/SoccerNet/videos/SoccerNet/italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/2_224p.mkv'
    else:
        input_video_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/SoccerNet/videos/SoccerNet/spain_laliga/2019-2020/2020-02-16 - 18-00 Real Madrid 2 - 2 Celta Vigo/2_224p.mkv'
    stream_infer = MultiAssistantStreamInference(model_config, ckpt_path, weight_dir_prefix, infer_config, generate_config, talker_config)
    # load response head
    response_head_ckpt = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251109-061114_strategy2_16/final'
    response_head_ckpt = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251111-090102_20251109-061114_-4_strategy2/checkpoint-846'
    stream_infer.model.load_response_head(response_head_ckpt)
    logger.info("Multi-assistant streaming inference model initialized.")
    stream_infer.new_session(task=data_task)
    # loop for inference, load video data as needed
    print(f"Starting inference on video: {input_video_path}.")
    input_video_path = get_video_path_from_local_or_hf(input_video_path, ckpt_path)
    infer(stream_infer, input_video_path, begin_time=0, duration=60)
