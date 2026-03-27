from torch.utils.data import Dataset
import os
import json
import math
from PIL import Image
import torch
from typing import Optional, List, Dict, Any, Union
import re


from dataclasses import dataclass
from src.utils.constants import (DEFAULT_ASSISTANT_ROLE_TOKEN,
                       DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                       DEFAULT_SYSTEM_ROLE_TOKEN, DEFAULT_USER_ROLE_TOKEN,
                       IGNORE_INDEX,
                       DEFAULT_VISION_END_TOKEN, DEFAULT_VISION_BEGIN_TOKEN, DEFAULT_VISION_TOKEN)
import random
from torch.nn.utils.rnn import pad_sequence
from src.utils.conversations import SYSTEM_PROMPT_MAP
from src.utils.conversations import BASE_PROMPT, PREFIX_PROMPT_MAP, SOLO_COMMENTARY_PROMPTS, MULTI_COMMENTARY_PROMPTS, GUIDANCE_COMMENTARY_PROMPTS, LIVECC_SYSTEM_PROMPT, EGO4D_SYSTEM_PROMPT

from src.utils.proact_process import process_interleave_mm_info
from qwen_vl_utils import process_vision_info as process_vision_info_vl
from src.utils.utils import _get, _squeeze_batch1, to_model_dtype
import logging
from datetime import datetime
import random
import gc

TRAIN_ANN_PATH_MAP = {
    'baldurs_gate_3': 'ann/baldurs_gate_3_final_train.jsonl',
    'csgo': 'ann/csgo_final_train.jsonl',
    'cyberpunk_2077': 'ann/cyberpunk_2077_final_train.jsonl',
    'elden_ring': 'ann/elden_ring_final_train.jsonl',
    'lol': 'ann/lol_final_train.jsonl',
    'minecraft': 'ann/minecraft_final_train.jsonl',
    'starcraft2': 'ann/starcraft2_final_train.jsonl',
    'streetfighter6': 'ann/streetfighter6_final_train.jsonl',
    'tears_of_the_kingdom': 'ann/tears_of_the_kingdom_final_train.jsonl',
    'yu_gi_oh': 'ann/yu_gi_oh_final_train.jsonl',
    'livecc': 'ann/livecc_final_train.jsonl',
    'ego4d': 'ann/ego4d_final_train.jsonl',
}

VAL_ANN_PATH_MAP = {
    'baldurs_gate_3': 'ann/baldurs_gate_3_final_val.jsonl',
    'csgo': 'ann/csgo_final_val.jsonl',
    'cyberpunk_2077': 'ann/cyberpunk_2077_final_val.jsonl',
    'elden_ring': 'ann/elden_ring_final_val.jsonl',
    'lol': 'ann/lol_final_val.jsonl',
    'minecraft': 'ann/minecraft_final_val.jsonl',
    'starcraft2': 'ann/starcraft2_final_val.jsonl',
    'streetfighter6': 'ann/streetfighter6_final_val.jsonl',
    'tears_of_the_kingdom': 'ann/tears_of_the_kingdom_final_val.jsonl',
    'yu_gi_oh': 'ann/yu_gi_oh_final_val.jsonl',
    'livecc': 'ann/livecc_final_val.jsonl',
    'ego4d': 'ann/ego4d_final_val.jsonl',
}

# VIDEO_DIR_MAP = {
#     'baldurs_gate_3': 'game_commentary/Baldurs_Gate_3/videos',
#     'csgo': 'game_commentary/CSGO/videos',
#     'cyberpunk_2077': 'game_commentary/Cyberpunk_2077/videos',
#     'elden_ring': 'game_commentary/Elden_Ring/videos',
#     'lol': 'game_commentary/LOL/videos',
#     'minecraft': 'game_commentary/Minecraft/videos',
#     'starcraft2': 'game_commentary/Starcraft2/videos',
#     'streetfighter6': 'game_commentary/Streetfighter6/videos',
#     'tears_of_the_kingdom': 'game_commentary/Tears_of_the_Kingdom/videos',
#     'yu_gi_oh': 'game_commentary/Yu_Gi_Oh/videos',
#     'livecc': 'live_sft/videos',
#     'ego4d_goal_step': 'ego4d/v2/full_scale',
# }

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

MIN_PIXELS = 128*28*28
MAX_PIXELS = 540*28*28
# MAX_VIDEO_PIXELS = 24576*28*28
MAX_VIDEO_PIXELS = 36*540*28*28

def drop_comments_randomly(anns, delete_prob=0.5):

    # 随机删除部分assistant的评论，delete_prob为删除概率
    if delete_prob <= 0 or delete_prob >= 1:
        return anns
    anns_new = []
    for ann in anns:
        if random.random() < delete_prob:
            continue
        anns_new.append(ann)
    return anns_new

# for video
@dataclass
class SupervisedStreamDatasetSample:
    input_ids: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    active_labels: torch.Tensor

    # optional
    video_second_per_grid: torch.Tensor = None
    second_per_grid_ts: torch.Tensor = None
    input_features: torch.Tensor = None
    feature_attention_mask: torch.Tensor = None

def construct_conversation_prompt(ann, active_eos_token='', silence_eos_token='', chunk_flag='<|FLAG|>'):
    # print(ann)
    video_begin_time = int(ann['video_begin'])
    video_end_time = int(ann['video_end'])
    # video_dir_path = ann['video_dir_path']
    cur_conversation = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": ann['system_prompt']
            }
        ]
    }]
    user_conversations = []
    for i in range(video_begin_time, video_end_time):
        cur_conversation.append({
            "role": "user",
            "content": [
                # {
                #     "type": "text",
                #     "text": f"Video chunk from {i} seconds to {i+1} seconds."
                # },
                {
                    "type": "video",
                    "video": ann['video_path'],
                    'video_start': video_begin_time,
                    'video_end': video_end_time,
                    'nframes': 2*int(video_end_time - video_begin_time),
                    "chunk_start": i,
                    "chunk_end": i + 1,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": min(MAX_PIXELS, MAX_VIDEO_PIXELS // (video_end_time - video_begin_time)),
                }
            ]
        })
        cur_conversation.append({
            "role": "assistant",
            "content": []
        })
    if 'history' in ann and ann['history'] is not None and ann['history'].strip() != '':
        # 如果存在整体history，添加到第一个user的history中
        cur_conversation[1]['content'].insert(0, {
            "type": "text",
            "text": ann['history']
        })
    for one_ann in reversed(ann['annotations']):
        if one_ann['end'] > video_end_time or one_ann['start'] < video_begin_time:
            continue
        # 三种类型，一个是assistant，直接放到assistant中

        if one_ann['role'] == 'assistant':
            if 'text' in one_ann and one_ann['text'] != '':
                begin_idx = one_ann['start'] - video_begin_time
                cur_conversation[begin_idx*2 + 2]['content'].append({
                    "type": "text",
                    "text": " " + one_ann['text'] + active_eos_token if one_ann['text'][-1] not in ['.', '!', '?'] else ' ' + one_ann['text']
                })
            else:
                raise ValueError(f"Assistant annotation must have 'text' or 'commentary' field: {one_ann}")
        # 第二种类型是user
        elif one_ann['role'] == 'user':
            if 'text' in one_ann and one_ann['text'] is not None and one_ann['text'].strip() != '' and one_ann['speaker'] != 'user':
                # 注意，可能存在多个history，即一个历史history和前一秒的history，这里倒序遍历每次放在第一个
                begin_idx = one_ann['start'] - video_begin_time
                to_append_text = f'[{one_ann["speaker"]}]: {one_ann["text"]}'
                if to_append_text[-1] not in ['.', '!', '?']:
                    to_append_text += active_eos_token
                # 如果存在history，则添加到text字段
                if cur_conversation[begin_idx*2 + 1]['content'][0]['type'] == 'text':
                    cur_conversation[begin_idx*2 + 1]['content'][0]['text'] = cur_conversation[begin_idx*2 + 1]['content'][0]['text'] + ' ' + to_append_text
                else:
                    cur_conversation[begin_idx*2 + 1]['content'].insert(0, {
                        "type": "text",
                        "text": to_append_text
                    })
            # user query
            elif 'query' in one_ann and one_ann['query'] is not None and one_ann['query'].strip() != '' and one_ann['speaker'] == 'user':
                begin_idx = one_ann['start'] - video_begin_time
                cur_conversation[begin_idx*2 + 1]['content'].append({
                    "type": "text",
                    "text": one_ann['query']
                })
            else:
                raise ValueError(f"User annotation must have 'text' or 'query' field: {one_ann}")

        else:
            raise ValueError(f"Invalid role in annotation: {one_ann['role']}")

    # cur_conversation = [conv for conv in cur_conversation if not (conv['role'] == 'assistant' and len(conv['content']) == 0)]
    for conv in cur_conversation:
        if conv['role'] == 'assistant':
            # 如果没有内容，填充[SILENCE]
            if len(conv['content']) == 0:
                conv['content'].append({
                    "type": "text",
                    # "text": "<|SILENCE|>"
                    "text": silence_eos_token
                })
        elif conv['role'] == 'user':
            if conv['content'][0]['type'] != 'video':
                # 添加'<|history_start|>', '<|history_end|>'
                conv['content'][0]['text'] = '<|history_start|>' + conv['content'][0]['text'] + '<|history_end|>'
            if conv['content'][-1]['type'] != 'video':
                # 添加'<|query_start|>', '<|query_end|>'
                conv['content'][-1]['text'] = '<|query_start|>' + conv['content'][-1]['text'] + '<|query_end|>'
    # 在每一个user后添加flag
    for i in range(len(cur_conversation)):
        if cur_conversation[i]['role'] == 'user':
            cur_conversation[i]['content'].append({
                "type": "text",
                "text": f"{chunk_flag}"
            })
    return cur_conversation


def construct_val_system_prompt(dataset_name, tag, persona):
    # 消除随机性
    if dataset_name in ['livecc']:
        return LIVECC_SYSTEM_PROMPT
    elif dataset_name in ['ego4d', 'Ego4D', 'ego4d_goal_step'] or tag in ['ego4d', 'Ego4D', 'ego4d_goal_step']:
        return EGO4D_SYSTEM_PROMPT
    else:
        if tag == 'Solo commentators':
            task_prompt = SOLO_COMMENTARY_PROMPTS[0]
        elif tag == 'Multiple commentators':
            task_prompt = MULTI_COMMENTARY_PROMPTS[0]
        elif tag == 'Guidance':
            task_prompt = GUIDANCE_COMMENTARY_PROMPTS[0]
        elif tag in ['SoccerNet', 'soccernet']:
            task_prompt = SOLO_COMMENTARY_PROMPTS[0]
        elif tag == 'Wukong':
            task_prompt = SOLO_COMMENTARY_PROMPTS[0]
        else:
            raise ValueError(f"Invalid tag: {tag}")
    prefix_prompt = PREFIX_PROMPT_MAP[dataset_name]
    system_prompt = " ".join([prefix_prompt, f'Here is the persona of the commentator:\n{persona}\n', task_prompt])
    return system_prompt

def construct_system_prompt(dataset_name, tag, persona):
    if random.random() < 0.1:
        return BASE_PROMPT
    if dataset_name in ['livecc']:
        return LIVECC_SYSTEM_PROMPT
    elif dataset_name in ['ego4d']:
        return EGO4D_SYSTEM_PROMPT
    else:
        if tag == 'Solo commentators':
            task_prompt = random.choice(SOLO_COMMENTARY_PROMPTS)
        elif tag == 'Multiple commentators':
            task_prompt = random.choice(MULTI_COMMENTARY_PROMPTS)
        elif tag == 'Guidance':
            task_prompt = random.choice(GUIDANCE_COMMENTARY_PROMPTS)
        else:
            raise ValueError(f"Invalid tag: {tag}")
    prefix_prompt = PREFIX_PROMPT_MAP[dataset_name]
    system_prompt = " ".join([prefix_prompt, f'Here is the persona of the commentator:\n{persona}\n', task_prompt])
    return system_prompt

class CustomCommentaryDataset(Dataset):
    def __init__(self, dataset_names: List[str], data_dir_path: List[str],
        processor, use_audio_in_video, is_train=True, 
        commentary_drop_rate=0.0, data_percentage=1.0, 
        min_pixels=None, max_pixels=None,
        active_eos_token=' ...', silence_eos_token=' ...', chunk_flag='<|FLAG|>',
    ):
        super().__init__()
        self.processor = processor
        # self.video_dir_path = data_dir_path
        # self.ann_path = ann_path
        self.use_audio_in_video = use_audio_in_video
        self.data_type = torch.bfloat16

        # self.min_pixels = min_pixels if min_pixels is not None else MIN_PIXELS
        # self.max_pixels = max_pixels if max_pixels is not None else MAX_PIXELS
        self.ann_list = []
        self.conversation_info_list = []

        self.commentary_drop_rate = commentary_drop_rate
        logger.info(f'commentary_drop_rate: {self.commentary_drop_rate}')
        self.data_percentage = data_percentage
        self.active_eos_token = active_eos_token if active_eos_token else ''
        self.silence_eos_token = silence_eos_token if silence_eos_token else ' ...'
        self.chunk_flag = '<|FLAG|>' if chunk_flag is None else chunk_flag
        self.im_start_token = DEFAULT_IM_START_TOKEN
        self.im_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.im_start_token)
        self.im_end_token = DEFAULT_IM_END_TOKEN
        self.im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.im_end_token)
        self.ignore_index = IGNORE_INDEX

        # ann_path = [ANN_MAP[dataset_name] for dataset_name in dataset_names]
        for idx, dataset_name in enumerate(dataset_names):
            # ann_path = ANN_MAP[dataset_name]
            if is_train:
                ann_path = os.path.join(data_dir_path, TRAIN_ANN_PATH_MAP[dataset_name])
            else:
                ann_path = os.path.join(data_dir_path, VAL_ANN_PATH_MAP[dataset_name])
            ann_list = []
            with open(ann_path, 'r') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        # ann['video_dir_path'] = self.video_dir_path[idx]
                        ann['video_dir_path'] = data_dir_path
                        ann['video_path'] = os.path.join(ann['video_dir_path'], ann['video_path'])
                        # dataset_name = dataset_names[idx]
                        # system_prompt_idx = random.randint(0,4)
                        # print(system_prompt_idx)
                        # ann['system_prompt'] = SYSTEM_PROMPT_MAP[dataset_name][system_prompt_idx]
                        ann['system_prompt'] = construct_system_prompt(dataset_name, ann['metadata']['tag'], ann['active_speaker']['persona'])
                        ann_list.append(ann)
                print(f'[{dataset_names[idx]}] Training dataset uses {len(ann_list)} samples.')
            self.ann_list.append(ann_list)
        
        self.ann_list = [ann for sublist in self.ann_list for ann in sublist]
        if is_train:
            print(f'Total training dataset uses {len(self.ann_list)} samples.')
        else:
            print(f'Total validation dataset uses {len(self.ann_list)} samples.')

        if data_percentage < 1.0:
            self.ann_list = self.ann_list[:int(len(self.ann_list)*data_percentage)]
            logger.info(f'Use {data_percentage*100}% of data, total {len(self.ann_list)} samples.')
        
        print(f'Load {len(self.ann_list)} samples.')

    
    '''对于active labels来说，
    <|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n<|im_start|>assistant\n 为response
    <|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n为silence
    '''
    def prepare_labels_for_multimodal(self,
                                    input_ids: torch.Tensor,) -> torch.Tensor:
        labels = input_ids.clone()
        labels_active = torch.zeros_like(labels).fill_(IGNORE_INDEX)
        input_len = len(input_ids[0])


        # im_start_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN))[1]
        # im_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN))[1]
        # # vision_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_END_TOKEN))[1]
        # # vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_BEGIN_TOKEN)
        # im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # vision_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_TOKEN)
        im_start_index = torch.where(labels == self.im_start_token_id)[1]
        im_end_index = torch.where(labels == self.im_end_token_id)[1]
        # im_end_token_id = self.im_end_token_id
        # vision_token_id = self.vision_token_id

        last_response_begin = 0
        last_response_end = 0
        last_response_length = 0
        for i in range(len(im_start_index)):
            im_start_idx = im_start_index[i].item()
            im_end_idx = im_end_index[i].item()
            if im_start_idx >= im_end_idx:
                raise ValueError(f"🙅 Invalid start and end token indices: {im_start_idx}, {im_end_idx}")
            else:
                cur_role = self.processor.tokenizer.convert_ids_to_tokens(labels[0][im_start_idx + 1].item())
                if cur_role == DEFAULT_SYSTEM_ROLE_TOKEN:
                    # label处理
                    # <|im_start|>system\nYou are a professional sports commentary Please given comment on the given video.<|im_end|>\n
                    labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    # active label处理
                    labels_active[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                elif cur_role == DEFAULT_USER_ROLE_TOKEN:
                    # <|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><chunk_flag><|im_end|>\n
                    labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    # active label处理
                    labels_active[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    labels_active[0][im_end_idx-1] = 0  # 初始化active label为0
                    # print(self.processor.tokenizer.decode(input_ids[0][im_start_idx:im_end_idx+2]))
                    # print(self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_idx-1].item()))
                    assert self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_idx-1].item()) == self.chunk_flag, \
                        f"🙅 The token before im_end must be <|im_end|>, but got {self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_idx-1].item())}"
                elif cur_role == DEFAULT_ASSISTANT_ROLE_TOKEN:
                    # <|im_start|>assistant\nfirmly by Flanagan<|im_end|>\n
                    # <|im_start|>assistant\n
                    labels[0][im_start_idx:im_start_idx + 3] = IGNORE_INDEX
                    # labels[0][im_end_idx] = IGNORE_INDEX
                    # \n
                    labels[0][im_end_idx+1] = IGNORE_INDEX

                    labels_active[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    if input_ids[0][im_start_idx+3].item() == self.processor.tokenizer.encode(self.silence_eos_token)[0]:
                        labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    else:
                        labels_active[0][im_end_index[i-1].item()-1] = 1
                    # 如果assistant中内容为空，将前一个user的active label设置为1
                    # if im_end_idx != im_start_idx + 3:
                    #     labels_active[0][im_end_index[i-1].item()] = 1
                else:
                    raise ValueError(f"🙅 Invalid role token: {cur_role}")
        return labels, labels_active
    
    def __len__(self):
        return len(self.ann_list)
        # return len(self.conversation_info_list)

    def prepare_inputs_for_qwen2_5_vl(self, conversation):
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        # audios, images, videos = process_interleave_mm_info(conversation, self.use_audio_in_video, return_video_kwargs=False)
        images, videos, video_kwargs = process_vision_info_vl(conversation[1:2], return_video_kwargs=True)
        videos = videos[0]
        videos = [videos[i:i+2] for i in range(0, len(videos), 2)]
        # min_pixels = self.min_pixels
        # max_pixels = min(self.max_pixels, MAX_VIDEO_PIXELS / len(videos))
                    # "min_pixels": MIN_PIXELS,
                    # "max_pixels": min(MAX_PIXELS, MAX_VIDEO_PIXELS // (video_end_time - video_begin_time))
        min_pixels = MIN_PIXELS
        max_pixels = min(MAX_PIXELS, MAX_VIDEO_PIXELS // len(videos))
        size = {
            'shortest_edge': min_pixels,
            'longest_edge': max_pixels,
        }
        inputs = self.processor(
            text=text, 
            images=images, 
            videos=videos, 
            return_tensors="pt",
            padding=True, 
            size=size,
        )
        return inputs

    def prepare_inputs_for_qwen2_vl(self, conversation):
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        # audios, images, videos = process_interleave_mm_info(conversation, self.use_audio_in_video, return_video_kwargs=False)
        images, videos, video_kwargs = process_vision_info_vl(conversation[1:2], return_video_kwargs=True)
        videos = videos[0]
        videos = [videos[i:i+2] for i in range(0, len(videos), 2)]
        # min_pixels = self.min_pixels
        # max_pixels = min(self.max_pixels, MAX_VIDEO_PIXELS / len(videos))
        min_pixels = MIN_PIXELS
        max_pixels = min(MAX_PIXELS, MAX_VIDEO_PIXELS // len(videos))
        size = {
            'shortest_edge': min_pixels,
            'longest_edge': max_pixels,
        }
        inputs = self.processor(
            text=text, 
            images=images, 
            videos=videos, 
            return_tensors="pt",
            padding=True, 
            size=size,
        )
        return inputs

    def prepare_inputs_for_qwen2_5_omni(self, conversation):
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        # audios, images, videos = process_interleave_mm_info(conversation, self.use_audio_in_video, return_video_kwargs=False)
        audios, images, videos = process_mm_info(conversation[1:2], use_audio_in_video=USE_AUDIO_IN_VIDEO)
        videos = [videos[i:i+2] for i in range(0, len(videos), 2)]
        # min_pixels = self.min_pixels
        # max_pixels = min(self.max_pixels, MAX_VIDEO_PIXELS / len(videos))
        min_pixels = MIN_PIXELS
        max_pixels = min(MAX_PIXELS, MAX_VIDEO_PIXELS // len(videos))
        size = {
            'shortest_edge': min_pixels,
            'longest_edge': max_pixels,
        }
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt",
            padding=True, 
            use_audio_in_video=self.use_audio_in_video,
            size=size,
        )
        if not self.use_audio_in_video:
            # 如果不使用视频中的音频，则将input_features和feature_attention_mask设为None
            inputs['input_features'] = None
            inputs['feature_attention_mask'] = None
        return inputs

    def prepare_inputs_for_qwen3_vl(self, conversation):
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        # 仅传入第一个video chunk获取完整的video信息
        first_user_conversation = conversation[1:2]
        images, videos, video_kwargs = process_vision_info_vl(first_user_conversation, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        chunk_length = len(videos[0]) // 2
        videos = [videos[0][i:i+2] for i in range(0, len(videos[0]), 2)]
        video_metadatas = [{
            'fps': video_metadatas[0]['fps'],
            'frames_indices': video_metadatas[0]['frames_indices'][i*2:i*2+2],
            'total_num_frames': video_metadatas[0]['total_num_frames'],
            'video_backend': video_metadatas[0]['video_backend'],
        } for i in range(chunk_length)]
        # since qwen-vl-utils has resize the images/videos, \
        # we should pass do_resize=False to avoid duplicate operation in processor!
        inputs = self.processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)

        if not self.use_audio_in_video:
            inputs['input_features'] = None
            inputs['feature_attention_mask'] = None
        return inputs

    def __getitem__(self, index):
        ann = self.ann_list[index]
        cur_conversation = construct_conversation_prompt(ann, active_eos_token=self.active_eos_token, silence_eos_token=self.silence_eos_token, chunk_flag=self.chunk_flag)
        if self.processor.__class__.__name__ == 'Qwen2_5OmniProcessor':
            inputs = self.prepare_inputs_for_qwen2_5_omni(cur_conversation)
        elif self.processor.__class__.__name__ == 'Qwen3VLProcessor':
            inputs = self.prepare_inputs_for_qwen3_vl(cur_conversation)
        elif self.processor.__class__.__name__ == 'Qwen2_5_VLProcessor':
            inputs = self.prepare_inputs_for_qwen2_5_vl(cur_conversation)
        elif self.processor.__class__.__name__ == 'Qwen2VLProcessor':
            inputs = self.prepare_inputs_for_qwen2_vl(cur_conversation)
        else:
            raise ValueError(f"Unknown processor class: {self.processor.__class__.__name__}")

        labels, active_labels = self.prepare_labels_for_multimodal(inputs['input_ids'])
        return SupervisedStreamDatasetSample(
            **inputs,
            labels=labels,
            active_labels=active_labels,
        )


# FIXME 这个collate fn可能有bug，之后需要仔细查一下。
class DataCollatorForStream2Text(object):
    """Collate examples for Stream2Text supervised fine-tuning."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, samples):
        # print(f"Collating {len(samples)} samples")
        input_ids_list = [sample.input_ids.squeeze(0) for sample in samples]
        attention_mask_list = [sample.attention_mask.squeeze(0) for sample in samples]
        pixel_values_videos_list = [sample.pixel_values_videos.squeeze(0) for sample in samples]
        video_grid_thw_list = [sample.video_grid_thw.squeeze(0) for sample in samples]
        
        labels_list = [sample.labels.squeeze(0) for sample in samples]
        active_labels_list = [sample.active_labels.squeeze(0) for sample in samples]
        
        pad_token = getattr(self.tokenizer, "pad_token", '[PAD]')
        _batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(pad_token))
        _batch_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        _batch_pixel_values_videos = torch.cat(pixel_values_videos_list, dim=0)
        _batch_video_grid_thw = torch.cat(video_grid_thw_list, dim=0)
        
        _batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        _batch_active_labels = pad_sequence(active_labels_list, batch_first=True, padding_value=IGNORE_INDEX)

        del input_ids_list, attention_mask_list, pixel_values_videos_list, video_grid_thw_list, labels_list, active_labels_list
        gc.collect()
        
        # return None
        # only for qwen2_5_omni
        _batch_video_second_per_grid = None
        if samples[0].video_second_per_grid is not None:
            video_second_per_grid_list = [sample.video_second_per_grid.squeeze(0) for sample in samples]
            _batch_video_second_per_grid = torch.cat(video_second_per_grid_list, dim=0)
        # only for qwen2_5_vl
        _batch_second_per_grid_ts = None
        if samples[0].second_per_grid_ts is not None:
            second_per_grid_ts_list = [sample.second_per_grid_ts.squeeze(0) for sample in samples]
            _batch_second_per_grid_ts = torch.cat(second_per_grid_ts_list, dim=0)
        # 如果包含input_features和feature_attention_mask，则进行pad和cat
        _batch_input_features = None
        _batch_feature_attention_mask = None
        if samples[0].input_features is not None and samples[0].feature_attention_mask is not None:
            input_features_list = [sample.input_features.squeeze(0) for sample in samples]
            feature_attention_mask_list = [sample.feature_attention_mask.squeeze(0) for sample in samples]
            _batch_input_features = torch.cat(input_features_list, dim=0)
            _batch_feature_attention_mask = torch.cat(feature_attention_mask_list, dim=0)

        # FIXME 加速数据读取，如果数据类型是float32，则转换为bfloat16
        if _batch_pixel_values_videos.dtype == torch.float32:
            _batch_pixel_values_videos = _batch_pixel_values_videos.to(torch.bfloat16)
        if _batch_input_features is not None and _batch_input_features.dtype == torch.float32:
            _batch_input_features = _batch_input_features.to(torch.bfloat16)
            
        to_return = {
            'input_ids': _batch_input_ids,
            'pixel_values_videos': _batch_pixel_values_videos,
            'video_grid_thw': _batch_video_grid_thw,
            'attention_mask': _batch_attention_mask,
            'labels': _batch_labels,
            'active_labels': _batch_active_labels,
        }
        if _batch_video_second_per_grid is not None:
            to_return['video_second_per_grid'] = _batch_video_second_per_grid
        if _batch_second_per_grid_ts is not None:
            to_return['second_per_grid_ts'] = _batch_second_per_grid_ts
        if _batch_input_features is not None:
            to_return['input_features'] = _batch_input_features
            to_return['feature_attention_mask'] = _batch_feature_attention_mask
        return to_return
    
if __name__ == '__main__':
    import os
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    from tqdm import tqdm
    processor = AutoProcessor.from_pretrained('chenjoya/LiveCC-7B-Base')
    processor.tokenizer.add_tokens(['<|elongated|>', '<|short_break|>', '<|long_break|>', '<|laugh|>'])
    processor.tokenizer.add_special_tokens({
        "additional_special_tokens": ['<|query_start|>', '<|query_end|>', '<|history_start|>', '<|history_end|>', '<|FLAG|>']
    })
    train_dataset = CustomCommentaryDataset(
        dataset_names=['baldurs_gate_3', 'csgo', 'cyberpunk_2077', 'elden_ring', 'lol', 'minecraft', 'starcraft2', 'streetfighter6', 'tears_of_the_kingdom', 'yu_gi_oh', 'livecc', 'ego4d'],
        data_dir_path='/home/v-weicaiyan/ds/DATA',
        processor=processor,
        use_audio_in_video=False,
        is_train=True,
        chunk_flag='<|FLAG|>',
    )
    token_num = 0

    # 你机器 CPU 核心数自己调：比如 8/16/32
    num_workers = min(32, os.cpu_count() or 4)

    # 如果你在 Dataset 里返回的是一个对象/字典，默认 collate 可能会堆叠；
    # 我们只需要原样拿出来，所以写一个 identity collate。
    def collate_fn(batch):
        # batch 是长度=1 的 list（因为 batch_size=1）
        return batch[0]

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,            # 这里主要是CPU处理，没必要 pin
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,           # 可调大一点提高吞吐
        collate_fn=collate_fn,
    )
    token_num = 0
    for sample in tqdm(loader, total=len(train_dataset)):
        # sample.input_ids: [1, seq] 或 [seq]，看你 Dataset 返回
        input_ids = sample.input_ids
        if hasattr(input_ids, "shape"):
            # torch.Tensor
            tokens = input_ids.shape[-1]
        else:
            # list 等
            tokens = len(input_ids[-1]) if isinstance(input_ids, list) else len(input_ids)

        token_num += int(tokens)

    print(f'Total tokens in training dataset: {token_num}')