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

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
ENDC = '\033[0m'
def get_video_path_from_local_or_hf(input_path, ckpt_path):
    if input_path == 'default':
        video_default_path = "asset/2025MSI_T1_vs_GEN_game5.mp4"
        print(f'No input video path provided, using default video from hf dataset: {os.path.join(ckpt_path, video_default_path)}.')
        input_video_path = _maybe_hf(ckpt_path, video_default_path, revision=None)
        return input_video_path
    return input_path

# 传入上一轮response，生成这一轮的response
def infer_one_chunk(stream_infer, audios, images, videos, assistant_responses, begin_second, prefix, user_query=None, history=None):
    # user_query = ''
    # if (begin_second + 15) % 60 == 0 and begin_second <= 1600:
    #     user_query = f'Do you think which team will win?'
    #     print(f'{prefix} User query: {user_query}')

    next_responses, _ = stream_infer.infer_one_chunk(audios, images, videos, user_query, assistant_responses, begin_second, history=history)
    for assistant_id, resp in next_responses.items():
        if resp.active:
            print(f'{prefix} Assistant {assistant_id}, active: {GREEN}{resp.active}{ENDC}, score: {GREEN}{resp.score:.4f}{ENDC}, response: {BLUE}{resp.commentary}{ENDC}')
        else:
            print(f'{prefix} Assistant {assistant_id}, active: {RED}{resp.active}{ENDC}, score: {RED}{resp.score:.4f}{ENDC}, response: {BLUE}{resp.commentary}{ENDC}')
        stream_infer.assistants[assistant_id].monitor()
    return next_responses

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
    assistant_responses = None
    for i in range(duration):
        current_second = begin_time + i
        one_chunk_audios = audios[i:i+1] if audios is not None else None
        one_chunk_images = images[i:i+1] if images is not None else None
        one_chunk_videos = videos[i:i+1] if videos is not None else None
        prefix = f"[Sec: {current_second} to {current_second + 1}] "
        assistant_responses = infer_one_chunk(stream_infer, one_chunk_audios, one_chunk_images, one_chunk_videos, assistant_responses, current_second, prefix)
    stream_infer.post_audio_generation()
    logger.info("Inference loop finished.")

if __name__ == "__main__":
    if False:
        from src.utils.utils import debug
        debug(9501)

    model_task = 'csgo'
    data_task = 'csgo'
    ckpt_path = 'trainer_output/proact_all_fulltuning_base_liveccbase_final/final'
    weight_dir_prefix = ''
    # ckpt_path = 'oaaoaa/ai_comp'
    # weight_dir_prefix = 'final'
    # model config
    model_config = ProActConfig(
        model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct",
        enable_audio_output=False,
        state_threshold=0.4,
        add_special_tokens=False,
        active_layer_id=-2,
    )
    # infer config
    infer_config = {
        'use_audio_in_video': False,
        'max_kv_tokens': 16384,
        'assistant_num': 2,
        'enable_tts': False,
        'save_dir': './infer_output',
    }
    generate_config = {
        'do_sample': True,
        'max_new_tokens': 12,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.25,
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
    # if data_task == 'lol':
    #     # lol
    #     input_video_path = '/home/v-weicaiyan/ds/DATA/game_commentary/LOL/videos/2025MSI_G2_vs_FLY/2025MSI_G2_vs_FLY_game2.mp4'
    # elif data_task == 'soccer':
    #     # soccer
    #     input_video_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet/italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/2_224p.mkv'
    # else:
    #     input_video_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet/spain_laliga/2019-2020/2020-02-16 - 18-00 Real Madrid 2 - 2 Celta Vigo/2_224p.mkv'
    input_video_path = '/home/v-weicaiyan/ds/DATA/game_commentary/CSGO/videos/major2024_game_24.mp4'
    stream_infer = MultiAssistantStreamInference(model_config, ckpt_path, weight_dir_prefix, infer_config, generate_config, talker_config)

    logger.info("Multi-assistant streaming inference model initialized.")
    stream_infer.new_session()
    # loop for inference, load video data as needed
    logger.info(f"Starting inference on video: {input_video_path}.")
    # input_video_path = get_video_path_from_local_or_hf(input_video_path, ckpt_path)
    infer(stream_infer, input_video_path, begin_time=1200, duration=300)
