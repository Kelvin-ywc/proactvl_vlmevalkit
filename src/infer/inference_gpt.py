import openai
import argparse
import os
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer
import debugpy
from typing import List
import base64
from io import BytesIO
from datetime import datetime
from openai import OpenAI


CLOSE_SOURCE_MOEL_LIST = ['gpt-4.1', 'gpt-4o', 'gpt-4.1-nano', 'gpt-5-chat']
OPEN_SOURCE_MODEL_LIST = []
API_KEY = ''

def get_client(model_name, api_version):
    if model_name in CLOSE_SOURCE_MOEL_LIST:
        from azure.identity import get_bearer_token_provider, AzureCliCredential  
        from openai import AzureOpenAI 
        credential = AzureCliCredential() 
        token_provider = get_bearer_token_provider( 
            credential, 
            "https://cognitiveservices.azure.com/.default") 
        aoiclient = AzureOpenAI( 
            azure_endpoint=os.getenv("GPT_ENDPOINT"), 
            azure_ad_token_provider=token_provider, 
            api_version=api_version, 
            max_retries=5, 
        )
        return aoiclient
    elif model_name in CLOSE_SOURCE_MOEL_LIST:
        return OpenAI(api_key=API_KEY)
    else:
        raise ValueError(f"Model {model_name} is not supported for closed source inference.")

def prepare_model(args):
    pass

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    
    return contents

def get_video_frames(video_path: str, sec: int, fps: int = 2) -> List[np.array]:
    frame_list = []
    video = VideoFileClip(video_path)
    frame = video.get_frame(sec)
    image = Image.fromarray((frame).astype(np.uint8))
    frame_list.append(image)
    return frame_list


def get_video_frames_base64(video_path: str, sec: int, step: int) -> List[str]:
    frame_list_base64 = []
    video = VideoFileClip(video_path)

    # 提取 sec 和 sec+0.5 秒的帧
    # timestamps = [sec, sec + 0.5]
    timestamps = np.arange(sec, sec+step, 0.5)
    print(len(timestamps))

    for t in timestamps:
        frame = video.get_frame(t)
        image = Image.fromarray(frame.astype(np.uint8)).resize((224, 224))
        image.save(os.path.join('./tmp', f"frame_{t}.png"))
        # 转为 base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        frame_list_base64.append(img_base64)

    return frame_list_base64



def inference_one_video_using_openai_client(client, model_name, video_path):
    system_prompt = '''你是一个游戏解说助手，我会不断给你提供游戏视频每一秒的内容，我希望你生成对应这一秒的解说内容，由于时间跨度只有一秒，你解说的内容不能太多，解说内容可以不是完整的句子，但下一秒内容需要紧跟上一秒来解说，剩余部分可以由下一秒继续接着解说，同时，回答内容时请同时考虑前面解说的内容以确保解说内容的连贯性.
    '''
    ## parse video and commentary on the first 60 seconds.
    chunk_step = 2
    pre_response = ''
    for sec in range(0, 30, chunk_step):
        image_inputs = get_video_frames_base64(video_path, sec, chunk_step) # from sec to sec+1
        # print(image_inputs)
        begin_time = datetime.now()
        user_prompt = f'images 提供了video的第{sec}秒到{sec+chunk_step}秒的画面，{pre_response}为之前画面解说内容,首先，请分析是否需要进行解说，若需要，请紧跟之前解说内容进行补充，若不需要，则直接返回<SILENCE>。'
        # response = client.chat.completions.create(
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt, "images": image_inputs}
        #     ],
        #     max_tokens=20,
        #     model=model_name
        # )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_inputs[0]}"}
                        },
                                                {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_inputs[1]}"}
                        }
                    ]
                }
            ],
            max_tokens=200,
        )
        end_time = datetime.now()
        res = response.choices[0].message.content
        if res == '<SILENCE>':
            pre_response = ''
        else:
            pre_response += res
        print(f'Second from {sec} to {sec+1} (latent time: {end_time-begin_time} second.): {res}')
        print(f'Current pre_response: {pre_response}')
    ## 

def main(args):
    if args.model_name in CLOSE_SOURCE_MOEL_LIST:
        client = get_client(args.model_name, args.api_version)
        print(f"Using closed source model {args.model_name}: {client}")
        inference_one_video_using_openai_client(client, args.model_name, args.video_path)
    elif args.model_name in OPEN_SOURCE_MODEL_LIST:
        model = prepare_model(args)
        print(f'Using open-source model {args.model}: {model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-4o", help='Model name to use for real time inference')
    parser.add_argument('--api_version', type=str, default="2024-05-01-preview", help='API version to use for the OpenAI service')
    parser.add_argument('--video_path', type=str, default='/home/v-weicaiyan/amlt_project/workspace/AI-Commentary/dataset/process_video_ffmpeg/CN_TL_2/CN_TL_2_part5.mp4')
    args = parser.parse_args()
    main(args)