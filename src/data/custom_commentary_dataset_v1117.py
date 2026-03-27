from torch.utils.data import Dataset
import os
import json
import math
from PIL import Image
import torch
from typing import Optional, List, Dict, Any, Union
import re
from qwen_omni_utils.v2_5.vision_process import extract_vision_info, fetch_image, fetch_video, process_vision_info
from qwen_omni_utils.v2_5.audio_process import process_audio_info
from dataclasses import dataclass
from src.utils.constants import (DEFAULT_ASSISTANT_ROLE_TOKEN,
                       DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                       DEFAULT_SYSTEM_ROLE_TOKEN, DEFAULT_USER_ROLE_TOKEN,
                       IGNORE_INDEX,
                       DEFAULT_VISION_END_TOKEN, DEFAULT_VISION_BEGIN_TOKEN, DEFAULT_VISION_TOKEN)
import random
from torch.nn.utils.rnn import pad_sequence
from src.utils.conversations import SYSTEM_PROMPT_MAP
from src.utils.proact_process import get_audio_chunks_from_one_video, process_interleave_mm_info, process_interleave_vision_info
from src.utils.utils import _get, _squeeze_batch1, to_model_dtype
import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)
MIN_PIXELS = 128*28*28
MAX_PIXELS = 128*28*28
# token_response_count = 0
# token_total_count = 0

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
    input_features: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor

    attention_mask: torch.Tensor
    feature_attention_mask: torch.Tensor

    video_second_per_grid: torch.Tensor

    labels: torch.Tensor
    active_labels: torch.Tensor

def check_source(vision_infos):
    source = vision_infos[0].get('video', None)
    if source is None:
        return False
    for info in vision_infos[1:]:
        if info['video'] != source:
            return False
    return True


def split_text_into_segments(text, n_segments):
    # 将 text 按空格拆分成 n_segments 段，尽量均匀分配单词, 如果segments段数小于n_segments, 则只取实际段数,空格需要保留
    words = text.split()
    # mask = []
    total_words = len(words)
    if total_words == 0:
        return [""] * n_segments
    if n_segments <= 0:
        raise ValueError("n_segments must be greater than 0")
    if n_segments > total_words:
        n_segments = total_words
    base_size = total_words // n_segments
    remainder = total_words % n_segments
    segments = []
    start = 0
    for i in range(n_segments):
        end = start + base_size + (1 if i < remainder else 0)

        segments.append(' '.join(words[start:end]))
        start = end
    # print(f'split text: {text} into segments: {segments}')
    # print(f'split mask: {mask}')
    return segments

def construct_conversation_prompt(ann):
    # print(ann)
    video_begin_time = ann['video_begin']
    video_end_time = ann['video_end']
    video_dir_path = ann['video_dir_path']
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
                    "video": os.path.join(video_dir_path, ann['video_path']),
                    'video_start': video_begin_time,
                    'video_end': video_end_time,
                    'nframes': 2*int(video_end_time - video_begin_time),
                    "chunk_start": i,
                    "chunk_end": i + 1,
                }
            ]
        })
        cur_conversation.append({
            "role": "assistant",
            "content": []
        })
    for one_ann in reversed(ann['annotations']):
        if one_ann['end_time'] > video_end_time or one_ann['begin_time'] < video_begin_time:
            continue
        # 三种类型，一个是assistant，直接放到assistant中

        if one_ann['role'] == 'assistant':
            if 'text' in one_ann and one_ann['text'] != '':
                begin_idx = one_ann['begin_time'] - video_begin_time
                cur_conversation[begin_idx*2 + 2]['content'].append({
                    "type": "text",
                    "text": one_ann['text']
                })
            elif 'commentary' in one_ann and one_ann['commentary'] != '':
                begin_idx = one_ann['begin_time'] - video_begin_time
                cur_conversation[begin_idx*2 + 2]['content'].append({
                    "type": "text",
                    "text": one_ann['commentary']
                })
            else:
                raise ValueError(f"Assistant annotation must have 'text' or 'commentary' field: {one_ann}")
        # 第二种类型是user
        elif one_ann['role'] == 'user':
            if 'history' in one_ann and one_ann['history'] != '':
                # 注意，可能存在多个history，即一个历史history和前一秒的history，这里倒序遍历每次放在第一个
                begin_idx = one_ann['begin_time'] - video_begin_time
                cur_conversation[begin_idx*2 + 1]['content'].insert(0, {
                    "type": "text",
                    "text": one_ann['history']
                })

            if 'query' in one_ann and one_ann['query'] != '':
                begin_idx = one_ann['begin_time'] - video_begin_time
                cur_conversation[begin_idx*2 + 1]['content'].append({
                    "type": "text",
                    "text": one_ann['query']
                })

        else:
            raise ValueError(f"Invalid role in annotation: {one_ann['role']}")

    cur_conversation = [conv for conv in cur_conversation if not (conv['role'] == 'assistant' and len(conv['content']) == 0)]
    return cur_conversation

class CustomCommentaryDataset(Dataset):
    def __init__(self, dataset_names: List[str], video_dir_path: List[str], ann_path: List[str], processor, use_audio_in_video, is_train=True, video_clip_length=90, commentary_drop_rate=0.0, data_percentage=1.0, min_pixels=None, max_pixels=None):
        super().__init__()
        self.processor = processor
        self.video_dir_path = video_dir_path
        self.ann_path = ann_path
        self.use_audio_in_video = use_audio_in_video
        self.data_type = torch.bfloat16

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.ann_list = []
        self.conversation_info_list = []

        self.commentary_drop_rate = commentary_drop_rate
        logger.info(f'commentary_drop_rate: {self.commentary_drop_rate}')
        self.data_percentage = data_percentage
        print(f'==================self.ann_path======================: {self.ann_path}')
        for idx, ann_path in enumerate(self.ann_path):
            ann_list = []
            with open(ann_path, 'r') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        ann['video_dir_path'] = self.video_dir_path[idx]
                        dataset_name = dataset_names[idx]
                        system_prompt_idx = random.randint(0,4)
                        # system_prompt_idx = 4
                        # print(system_prompt_idx)
                        ann['system_prompt'] = SYSTEM_PROMPT_MAP[dataset_name][system_prompt_idx]
                        ann_list.append(ann)
            self.ann_list.append(ann_list)
        
        if is_train:
            # 每个个ann_list取前90%
            for i in range(len(self.ann_list)):
                ann_len = len(self.ann_list[i])
                self.ann_list[i] = self.ann_list[i][:int(ann_len*0.9)]
                logger.info(f'Training dataset {dataset_names[i]} uses {len(self.ann_list[i])} samples.')
            self.ann_list = [ann for sublist in self.ann_list for ann in sublist]

        else:
            for i in range(len(self.ann_list)):
                ann_len = len(self.ann_list[i])
                self.ann_list[i] = self.ann_list[i][int(ann_len*0.9):]
                logger.info(f'Validation dataset {dataset_names[i]} uses {len(self.ann_list[i])} samples.')
            self.ann_list = [ann for sublist in self.ann_list for ann in sublist]
        if data_percentage < 1.0:
            self.ann_list = self.ann_list[:int(len(self.ann_list)*data_percentage)]
            logger.info(f'Use {data_percentage*100}% of data, total {len(self.ann_list)} samples.')
        
        print(f'Load {len(self.ann_list)} samples from {self.ann_path}')

        # construct conversations
        # for ann in self.ann_list:
        #     conversation_info = {
        #         'ann': ann,
        #         # 'video_dir_path': self.video_dir_path,
        #     }
        #     self.conversation_info_list.append(conversation_info)

            # conversation = construct_conversation_prompt(ann, self.video_dir_path)
            # self.conversation_list.append(conversation)
    
    '''对于active labels来说，
    <|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n<|im_start|>assistant\n 为response
    <|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|im_end|>\n为silence
    '''
    def prepare_labels_for_multimodal(self,
                                    input_ids: torch.Tensor,) -> torch.Tensor:
        labels = input_ids.clone()
        labels_active = torch.zeros_like(labels).fill_(IGNORE_INDEX)
        input_len = len(input_ids[0])


        im_start_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN))[1]
        im_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN))[1]
        vision_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_END_TOKEN))[1]
        vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_BEGIN_TOKEN)
        im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        vision_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_TOKEN)

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
                    labels_active[0][im_end_idx] = 0  # 初始化active label为0
                    # print(self.processor.tokenizer.decode(input_ids[0][im_start_idx:im_end_idx+2]))
                    # print(self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_idx-1].item()))
                    assert self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_idx].item()) == '<|im_end|>', \
                        f"🙅 The token before im_end must be <|im_end|>, but got {self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_idx-1].item())}"
                elif cur_role == DEFAULT_ASSISTANT_ROLE_TOKEN:
                    # <|im_start|>assistant\nfirmly by Flanagan<|im_end|>\n
                    # <|im_start|>assistant\n
                    labels[0][im_start_idx:im_start_idx + 3] = IGNORE_INDEX
                    # labels[0][im_end_idx] = IGNORE_INDEX
                    # \n
                    labels[0][im_end_idx+1] = IGNORE_INDEX

                    labels_active[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    # 如果assistant中内容为空，将前一个user的active label设置为1
                    if im_end_idx != im_start_idx + 3:
                        labels_active[0][im_end_index[i-1].item()] = 1
                    try:
                        assert self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_index[i-1].item()].item()) == '<|im_end|>', \
                            f"🙅 The token before im_end must be <|im_end|>, but got {self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_end_index[i-1].item()-1].item())}"
                        # assert前一个role是user
                        assert self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_start_index[i-1].item() + 1].item()) == DEFAULT_USER_ROLE_TOKEN, \
                            f"🙅 The role before assistant must be user, but got {self.processor.tokenizer.convert_ids_to_tokens(input_ids[0][im_start_index[i-1].item() + 1].item())}"
                    except AssertionError as e:
                        print(self.processor.tokenizer.decode(input_ids[0][im_end_index[i-1].item()-1:im_start_idx + 3]))
                        print(f"Error: {e}")
                        # raise e
                else:
                    raise ValueError(f"🙅 Invalid role token: {cur_role}")
        
        # assert (torch.sum(labels_active == 1) + torch.sum(labels_active == 0)) == 90, f"🙅 The number of active labels (1) should be 90."
        # print(input_ids)
        # print(labels[0][torch.where(labels[0] != IGNORE_INDEX)])
        # print(self.processor.tokenizer.decode(labels[0][torch.where(labels[0] != IGNORE_INDEX)]))
        # print(labels_active)
        # exit(0)
        # global token_response_count, token_total_count

        # print(f'Current total response token count: {torch.sum(labels_active == 1).item()}, total token count: {torch.sum(labels_active != IGNORE_INDEX).item()}, response ratio: {torch.sum(labels_active == 1).item()/torch.sum(labels_active != IGNORE_INDEX).item():.4f}')
        return labels, labels_active
    
    def __len__(self):
        return len(self.ann_list)
        # return len(self.conversation_info_list)

    def __getitem__(self, index):
        # print(f'Fetching item {index}')
        # ann = self.ann_list[index]
        # cur_conversation = self.conversation_list[index]
        # conversation_info = self.conversation_info_list[index]
        ann = self.ann_list[index]
        # randomly drop some comments for data augmentation
        if False:
            conversation_info['ann']['annotations'] = drop_comments_randomly(ann['annotations'], delete_prob=self.commentary_drop_rate)
        # print(conversation_info['ann'])
        cur_conversation = construct_conversation_prompt(ann)
        if False:
            # 剔除所有assistant的评论
            cur_conversation = [c for c in cur_conversation if not (c['role'] == 'assistant')]
            # print(cur_conversation)
        # print(cur_conversation)
        text = self.processor.apply_chat_template(cur_conversation, add_generation_prompt=False, tokenize=False)

        audios, images, videos = process_interleave_mm_info(cur_conversation, self.use_audio_in_video, return_video_kwargs=False)
        # 保存 videos
        # torch.save(videos, f'/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/tmp/video.pt')
        # print(videos)
        # time2 = datetime.now()
        # print(f'Process multimodal info time: {(time2 - time1).total_seconds():.2f} seconds')
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt",
            padding=True, 
            use_audio_in_video=self.use_audio_in_video,
            min_pixels=self.min_pixels if self.min_pixels is not None else MIN_PIXELS,
            max_pixels=self.max_pixels if self.max_pixels is not None else MAX_PIXELS,
        )
        # time3 = datetime.now()
        # print(f'Processor time: {(time3 - time2).total_seconds():.2f} seconds')
        del audios
        del images
        del videos

        if not self.use_audio_in_video:
            # 如果不使用视频中的音频，则将input_features和feature_attention_mask设为None
            inputs['input_features'] = None
            inputs['feature_attention_mask'] = None
        
        labels, active_labels = self.prepare_labels_for_multimodal(inputs['input_ids'])
        # inputs = to_model_dtype(inputs, self.data_type)
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
        video_second_per_grid_list = [sample.video_second_per_grid.squeeze(0) for sample in samples]
        labels_list = [sample.labels.squeeze(0) for sample in samples]
        active_labels_list = [sample.active_labels.squeeze(0) for sample in samples]
        
        pad_token = getattr(self.tokenizer, "pad_token", '[PAD]')
        _batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(pad_token))
        _batch_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        # _batch_pixel_values_videos_mask = torch.stack(pixel_values_videos_list, dim=0)
        # print(pixel_values_videos_list[0].shape)
        # print(pixel_values_videos_list[1].shape)
        _batch_pixel_values_videos = torch.cat(pixel_values_videos_list, dim=0)
        # _batch_video_grid_thw = torch.stack(video_grid_thw_list, dim=0)
        _batch_video_grid_thw = torch.cat(video_grid_thw_list, dim=0)
        _batch_video_second_per_grid = torch.cat(video_second_per_grid_list, dim=0)
        _batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        _batch_active_labels = pad_sequence(active_labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        # return None

        del input_ids_list, attention_mask_list, pixel_values_videos_list, video_grid_thw_list, video_second_per_grid_list, labels_list, active_labels_list

        # 如果包含input_features和feature_attention_mask，则进行pad和cat
        _batch_input_features = None
        _batch_feature_attention_mask = None
        if samples[0].input_features is not None and samples[0].feature_attention_mask is not None:
            input_features_list = [sample.input_features.squeeze(0) for sample in samples]
            feature_attention_mask_list = [sample.feature_attention_mask.squeeze(0) for sample in samples]
            _batch_input_features = torch.cat(input_features_list, dim=0)
            _batch_feature_attention_mask = torch.cat(feature_attention_mask_list, dim=0)

        # FIXME 加速数据读取，如果数据类型是float32，则转换为bfloat16
        # if _batch_pixel_values_videos.dtype == torch.float32:
        #     _batch_pixel_values_videos = _batch_pixel_values_videos.to(torch.bfloat16)
        # if _batch_input_features is not None and _batch_input_features.dtype == torch.float32:
        #     _batch_input_features = _batch_input_features.to(torch.bfloat16)
        return {
            'input_ids': _batch_input_ids,
            'input_features': _batch_input_features,
            'pixel_values_videos': _batch_pixel_values_videos,
            'video_grid_thw': _batch_video_grid_thw,
            'attention_mask': _batch_attention_mask,
            'feature_attention_mask': _batch_feature_attention_mask,
            'video_second_per_grid': _batch_video_second_per_grid,
            'labels': _batch_labels,
            'active_labels': _batch_active_labels,
        }
    
    # input_ids: torch.Tensor
    # input_features: torch.Tensor
    # pixel_values_videos: torch.Tensor
    # video_grid_thw: torch.Tensor

    # attention_mask: torch.Tensor
    # feature_attention_mask: torch.Tensor

    # video_second_per_grid: torch.Tensor

    # labels: torch.Tensor
    # active_labels: torch.Tensor
if __name__ == '__main__':

    from transformers import Qwen2_5OmniProcessor
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import debugpy
    if True:
        try:
            debugpy.listen(('localhost',9501))
            print(f'debug listen on port 9501')
            debugpy.wait_for_client()
        except Exception as e:
            raise RuntimeError(f"Failed to start debugpy: {e}")
        
    processor = Qwen2_5OmniProcessor.from_pretrained(
        'Qwen/Qwen2.5-Omni-7B'
    )

    video_dir_path = ['/home/v-weicaiyan/amlt_project/DATA/live_sft/videos/videos', '/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet', '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos']
    ann_path = [
        '/home/v-weicaiyan/ds/DATA/ann/all_in_one/livecc_split_clips_60s_combined.jsonl',
        '/home/v-weicaiyan/ds/DATA/ann/all_in_one/soccer_split_clips_60s_overlap40s_combined.jsonl',
        '/home/v-weicaiyan/ds/DATA/ann/all_in_one/lol_split_clips_60s_overlap40s_combined.jsonl'
    ]

    # video_dir_path = [video_dir_path[1]]
    # ann_path = [ann_path[1]]
    dataset = CustomCommentaryDataset(
        dataset_names=['livecc', 'soccer', 'lol'],
        video_dir_path=video_dir_path,
        ann_path=ann_path,
        processor=processor,
        use_audio_in_video=False,
    )

    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    # exit(0)
    collator = DataCollatorForStream2Text(processor.tokenizer)
    # print(collator([dataset[1]]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=DataCollatorForStream2Text(processor.tokenizer), shuffle=True, num_workers=4)
    idx = 0
    # for data in dataset:
    #     idx += 1
    # print(f'Dataset length: {len(dataset)}, iter {idx}')
    # exit()
    for batch in dataloader:
        idx += 1
        print(f'Iter {idx}:')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f'{k}: {v.shape} {v.dtype} {v.device}')
            else:
                print(f'{k}: {type(v)}')
            # token_response_count, token_total_count
        # print(f'Current total response token count: {token_response_count}, total token count: {token_total_count}, response ratio: {token_response_count/token_total_count:.4f}')
    print(f'Finish one epoch, {idx} iters')
