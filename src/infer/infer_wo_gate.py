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
    next_responses = stream_infer.infer_one_chunk_wo_gate(audios, images, videos, assistant_responses, begin_second)
    for assistant_id, resp in next_responses.items():
        print(f'{prefix} Assistant {assistant_id} response: {resp.response}, active: {resp.active}, score: {resp.score:.4f}')
    return next_responses

def infer(stream_infer, input_video_path, duration):
    # first prime all assistants with system prompts
    stream_infer.prime_system_prompts()
    # process video info to get audios, images, videos
    video_info = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": input_video_path,
                    "video_start": 0,
                    "video_end": duration,
                    "nframes": duration * 2,
                }
            ]
        }
    ]
    audios, images, videos = process_interleave_mm_info(video_info, stream_infer.use_audio_in_video, return_video_kwargs=False)

    for i in range(duration):
        one_chunk_audios = audios[i:i+1] if audios is not None else None
        one_chunk_images = images[i:i+1] if images is not None else None
        one_chunk_videos = videos[i:i+1] if videos is not None else None
        prefix = f"[Sec: {i} to {i+1}] "
        assistant_responses = infer_one_chunk(stream_infer, one_chunk_audios, one_chunk_images, one_chunk_videos, None, i, prefix)
    # stream_infer.post_audio_generation()
    logger.info("Inference loop finished.")

if __name__ == "__main__":
    if True:
        from src.utils.utils import debug
        debug(9501)
    # configuration
    # ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251027-044755/checkpoint-462'
    ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/exp_amlt_20251028-054349/final'
    weight_dir_prefix = ''
    # model config
    model_config = ProActConfig(
        model_name_or_path = "Qwen/Qwen2.5-Omni-7B",
        enable_audio_output=False,
        state_threshold=0.2,
        add_special_tokens=False,
        active_layer_id=-1,
    )
    # infer config
    infer_config = {
        'use_audio_in_video': False,
        'max_kv_tokens': 3072,
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
    input_video_path = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_T1_vs_GEN/2025MSI_T1_vs_GEN_game5.mp4'
    stream_infer = MultiAssistantStreamInference(model_config, ckpt_path, weight_dir_prefix, infer_config, generate_config, talker_config)
    logger.info("Multi-assistant streaming inference model initialized.")
    stream_infer.new_session()
    # loop for inference, load video data as needed
    logger.info(f"Starting inference on video: {input_video_path}.")
    input_video_path = get_video_path_from_local_or_hf(input_video_path, ckpt_path)
    infer(stream_infer, input_video_path, duration=120)
