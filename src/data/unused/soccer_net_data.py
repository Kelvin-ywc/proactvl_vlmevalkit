from torch.utils.data import Dataset
import os
import json
import math
from PIL import Image
import torch
from typing import Optional
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


@dataclass
class SupervisedStreamDatasetSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    video_second_per_grid: torch.Tensor
    labels: torch.Tensor
    active_labels: torch.Tensor
    conversations: list

def split_annotations_by_time(annotations, video_path, segment_duration=90):
    """
    根据时间段切分注释列表，并为每个分段添加视频路径和起止时间。
    
    Args:
        annotations (list): 包含注释字典的列表。
        video_path (str): 视频文件的路径。
        segment_duration (int): 每个时间段的秒数，默认为 180 秒（3分钟）。
        
    Returns:
        list: 包含多个字典的列表，每个字典代表一个时间段，
              包含 'start_time', 'end_time', 'video_path' 和 'annotations' 列表。
    """
    if not annotations:
        return []

    # 找到总时长
    last_end_time = math.floor(max(ann['offset'] + ann['duration'] for ann in annotations))
    num_segments = last_end_time // segment_duration
    
    # 初始化结果列表，为每个时间段创建一个包含元数据的字典
    result = []
    for i in range(int(num_segments)):
        result.append({
            'video_path': video_path,
            'start_time': i * segment_duration,
            'end_time': (i + 1) * segment_duration,
            'annotations': []
        })
    
    for ann in annotations:
        start_time = ann['offset']
        end_time = ann['offset'] + ann['duration']
        ann = {
            'offset': ann['offset'],
            'duration': ann['duration'],
            'commentary': ann['commentary']
        }

        start_segment_index = math.floor(start_time / segment_duration)
        end_segment_index = math.floor(end_time / segment_duration)

        # 检查注释是否跨越时间段
        if (start_segment_index+1) * segment_duration >= end_time:
            if start_segment_index < len(result):
                result[start_segment_index]['annotations'].append(ann)
        else:
            # TODO 目前实现取0-1分钟，1-2分钟的视频片段，后续改进使用队列进行处理
            pass

    return result


def check_source(vision_infos):
    source = vision_infos[0].get('video', None)
    if source is None:
        return False
    for info in vision_infos[1:]:
        if info['video'] != source:
            return False
    return True

def process_interleave_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_interleave_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios,) + vision

def process_interleave_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    # if all the clip belongs to the same video, only read the video one time and segment
    flag_video_source = check_source(vision_infos)
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    if flag_video_source:
        video_input, video_sample_fps = fetch_video(vision_infos[0], return_video_sample_fps=True)
        total_frames = video_input.shape[0]
        begin_frame_idx = 0
        end_frame_idx = 0
        # video_inputs = [video_input[i] for i in range(video_input.shape[0])]
        video_inputs = [video_input[i:i+2] for i in range(0, len(video_input), 2)]
        video_sample_fps_list = [video_sample_fps] * len(video_inputs)
        # for vision_info in vision_infos:
        #     end_time = vision_info.get('chunk_end', None)
        #     end_frame_idx = math.ceil(end_time * video_sample_fps)
        #     video_sample_fps_list.append(video_sample_fps)
        #     video_inputs.append(video_input[begin_frame_idx:end_frame_idx])
        #     print(f'Processing video segment: {begin_frame_idx} - {end_frame_idx-1}')
        #     begin_frame_idx = end_frame_idx
        # segment according to offset label

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info and not flag_video_source:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        elif not flag_video_source:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs

def replace_segment_with_chunk(texts, replace_list):
    """
    对 texts 中的每个字符串：
    将第 i 个 <|VIDEO|> 替换为 replace_list[i]。
    要求每个 text 里的 <|VIDEO|> 数量 == len(replace_list)
    """
    def replace_in_text(text):
        count = 0  # 每个 text 独立计数
        def repl(match):
            nonlocal count
            if count >= len(replace_list):
                raise ValueError("某个 text 中 <|VIDEO|> 数量超过 replace_list 长度")
            replacement = replace_list[count]
            count += 1
            return replacement

        new_text = re.sub(r"<\|vision_bos\|><\|VIDEO\|><\|vision_eos\|>", repl, text)

        if count != len(replace_list):
            raise ValueError("某个 text 中 <|VIDEO|> 数量与 replace_list 长度不一致")

        return new_text

    return [replace_in_text(text) for text in texts]



class SoccerNetCommentaryDataset(Dataset):
    def __init__(self, video_data_dir_path: str, ann_dir_path: str, processor, use_audio_in_video, video_clip_length):
        super().__init__()
        self.processor = processor
        self.video_data_dir_path = video_data_dir_path
        self.ann_dir_path = ann_dir_path
        self.ann_list = []
        self.conversation = []
        self.video_clip_length = video_clip_length
        for root, dirs, files in os.walk(self.ann_dir_path):
            for file in files:
                if file.endswith('.json'):
                    '''
                    file: europe_uefa-champions-league__2014-2015__2015-02-17_-_22-45_Paris_SG_1_-_1_Chelsea_2.json
                    eles[0]: europe_uefa-champions-league
                    eles[1]: 2014-2015
                    eles[2]: 2015-02-17_-_22-45_Paris_SG_1_-_1_Chelsea_2.json
                    ''' 
                    eles = file.split('__')

                    #FIXME hard code
                    ann_file_name = eles[-1][:-7].replace('_', ' ')
                    video_index = eles[-1][-6]
                    # cur_video_path: /home/v-weicaiyan/amlt_project/DATA/SoccerNet/europe_uefa-champions-league/2014-2015/2015-02-17_-_22-45_Paris_SG_1_-_1_Chelsea/2_224p.mkv
                    cur_video_path = os.path.join(video_data_dir_path, eles[0], eles[1], ann_file_name, f'{video_index}_224p.mkv')
                    ann_file_path = os.path.join(self.ann_dir_path, file)
                    with open(ann_file_path, 'r') as ann_file:
                        ann = json.load(ann_file)
                    
                    res = split_annotations_by_time(ann, cur_video_path, self.video_clip_length)

                    # constrain each video into 3 mins
                    self.ann_list.extend(res)
                    for one_sample in res:
                        cur_conversation = [
                                        {
                                "role": "system",
                                "content": [
                                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                                ],
                            },
                        ]
                        
                        for one_chunk in one_sample['annotations']:
                            cur_conversation.append({
                                "role": "user",
                                "content": [{
                                    "type": "video",
                                    "video": cur_video_path,
                                    'video_start': one_sample['start_time'],
                                    'video_end': one_sample['end_time'],
                                    'chunk_start': one_chunk['offset'],
                                    'chunk_end': one_chunk['offset'] + one_chunk['duration'],
                                }]
                            })
                            cur_conversation.append({
                                "role": "assistant",
                                "content": [{
                                    "type": "text",
                                    "text": one_chunk['commentary']
                                }]
                            })
                        if len(cur_conversation) > 2:
                            self.conversation.append(cur_conversation)
        if False:
            print(f'ann len: {len(self.conversation)}')
            print(f'sample: {self.conversation[0]}')
            exit()
            # print(f'sample: {self.ann_list[1]}')
        # print(f'Sample: {self.ann_list[0]}')
    
    def prepare_labels_for_multimodal(self,
                                    input_ids: torch.Tensor,) -> torch.Tensor:
        labels = input_ids.clone()
        labels_active = torch.zeros_like(labels).fill_(IGNORE_INDEX)
        input_len = len(input_ids[0])
        # mask audio token <|AUDIO|>
        # audio_token_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_token))
        # print(audio_token_mask)
        # labels = labels.masked_fill(audio_token_mask, IGNORE_INDEX)
        # mask system prompt
        im_start_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN))[1]
        im_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN))[1]
        vision_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_END_TOKEN))[1]
        vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_BEGIN_TOKEN)
        im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        vision_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VISION_TOKEN)
        for i in range(len(vision_end_index)):
            cur_vision_end_idx = vision_end_index[i].item()
            labels_active[0][cur_vision_end_idx] = 0

            if input_ids[0][cur_vision_end_idx + 1].item() == im_end_token_id:
                labels_active[0][cur_vision_end_idx] = 1
                # print(f'Active token idx: {cur_vision_end_idx}')

        for i in range(len(im_start_index)):
            im_start_idx = im_start_index[i].item()
            im_end_idx = im_end_index[i].item()
            if im_start_idx >= im_end_idx:
                raise ValueError(f"🙅 Invalid start and end token indices: {im_start_idx}, {im_end_idx}")
            else:
                cur_role = self.processor.tokenizer.convert_ids_to_tokens(labels[0][im_start_idx + 1].item())
                if cur_role == DEFAULT_SYSTEM_ROLE_TOKEN:
                    labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                elif cur_role == DEFAULT_USER_ROLE_TOKEN:
                    labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    # labels[0][im_start_idx:im_end_idx - 1] = IGNORE_INDEX
                    # labels[0][im_end_idx:im_end_idx + 2] = IGNORE_INDEX
                elif cur_role == DEFAULT_ASSISTANT_ROLE_TOKEN:
                    # if self.ignore_assistant_ans:
                        # labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    # else:
                    labels[0][im_start_idx:im_start_idx + 3] = IGNORE_INDEX
        
        # labels.masked_fill_(active_gen_mask, self.processor.tokenizer.convert_tokens_to_ids(ACTIVE_GEN_TOKEN))
        # labels.masked_fill_(silent_mask, self.processor.tokenizer.convert_tokens_to_ids(SILENT_TOKEN))
        
        return labels, labels_active
    
    def __len__(self):
        return len(self.conversation)


    def __getitem__(self, index):
        cur_conversation = self.conversation[index]
        # the text should be <system>: ...<user><VIDEO><AUDIO><VIDEO><AUDIO><VIDEO><AUDIO><asistant>: ...<user><VIDEO><AUDIO><VIDEO><AUDIO><VIDEO><AUDIO><asistant>: ...
        text = self.processor.apply_chat_template(cur_conversation, add_generation_prompt=False, tokenize=False)

        replace_cnt = []
        begin_sec = 0
        for item in cur_conversation[1:]:
            if item['role'] == 'user' and item['content'][0]['type'] == 'video':
                # print(math.ceil(item['content'][0]['chunk_end']))
                begin_sec = item['content'][0]['video_start']
                replace_cnt.append(math.ceil(item['content'][0]['chunk_end']))
        # vision_infos = extract_vision_info(cur_conversation)

        replace_list = ['<|vision_bos|><|VIDEO_CHUNK|><|vision_eos|>'*(replace_cnt[0]-begin_sec)]
        for i in range(len(replace_cnt)-1):
            replace_list.append('<|vision_bos|><|VIDEO_CHUNK|><|vision_eos|>' * (replace_cnt[i+1] - replace_cnt[i]))
        text = replace_segment_with_chunk(text, replace_list)
        text = [te.replace("<|vision_bos|><|VIDEO_CHUNK|><|vision_eos|>", "<|vision_bos|><|VIDEO|><|vision_eos|>") for te in text]
        audios, images, videos = process_interleave_mm_info(cur_conversation, False, return_video_kwargs=False)
        videos = videos[:replace_cnt[-1]-begin_sec]
        inputs = self.processor(text=text, 
                                audios=audios, 
                                images=images, 
                                videos=videos, 
                                return_tensors="pt",
                                padding=True, 
                                use_audio_in_video=False,
                                )
        # get video frames: [120, 3, 140, 140]
        input_ids = inputs['input_ids']
        # TODO
        labels, active_labels = self.prepare_labels_for_multimodal(input_ids)
        return SupervisedStreamDatasetSample(
            **inputs,
            labels=labels,
            active_labels=active_labels,
            conversations=cur_conversation,
            # audio_segment_pos=inputs['audio_segment_pos']
        )

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

        _batch_conversations = [sample.conversations for sample in samples]
        
        _batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
        _batch_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        # _batch_pixel_values_videos_mask = torch.stack(pixel_values_videos_list, dim=0)
        # print(pixel_values_videos_list[0].shape)
        # print(pixel_values_videos_list[1].shape)
        _batch_pixel_values_videos = torch.concat(pixel_values_videos_list, dim=0)
        # _batch_video_grid_thw = torch.stack(video_grid_thw_list, dim=0)
        _batch_video_grid_thw = torch.cat(video_grid_thw_list, dim=0)
        _batch_video_second_per_grid = torch.cat(video_second_per_grid_list, dim=0)
        _batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        _batch_active_labels = pad_sequence(active_labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        # return None
        return {
            'input_ids': _batch_input_ids,
            'attention_mask': _batch_attention_mask,
            'pixel_values_videos': _batch_pixel_values_videos,
            'video_grid_thw': _batch_video_grid_thw,
            'video_second_per_grid': _batch_video_second_per_grid,
            'labels': _batch_labels,
            'active_labels': _batch_active_labels,
            "conversations": _batch_conversations,
        }
    
if __name__ == '__main__':
    from transformers import Qwen2_5OmniProcessor
    import debugpy
    if False:
        try:
            debugpy.listen(('localhost',9501))
            print(f'debug listen on port 9501')
            debugpy.wait_for_client()
        except Exception as e:
            raise RuntimeError(f"Failed to start debugpy: {e}")
        
    processor = Qwen2_5OmniProcessor.from_pretrained(
        'Qwen/Qwen2.5-Omni-7B'
    )
    dataset = SoccerNetCommentaryDataset(
        video_data_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet',
        ann_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/commentaries',
        processor=processor,
        use_audio_in_video=False,
        video_clip_length=90
    )
    from dataclasses import asdict
    # print(dataset[28])
    # print(dataset[29])
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=DataCollatorForStream2Text(processor.tokenizer),
        num_workers=8
    )
    for batch in dataloader:
        # print(batch['input_ids'].dtype)
        # print(batch['pixel_values_videos'].dtype)
        for key, value in batch.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)}")
                print(f"data type: {value[0].dtype}")
            elif isinstance(value, torch.Tensor):
                print(f'{key}: {value.shape}')
                print(f"data type: {value.dtype}")
        # exit()
    # for idx, item in enumerate(dataset):
    #     assert item.video_grid_thw.shape[0] * 576 == item.pixel_values_videos.shape[0], (
    #         f"Mismatch: video_grid_thw.shape[0] * 576 = {item.video_grid_thw.shape[0] * 576}, "
    #         f"but pixel_values_videos.shape[0] = {item.pixel_values_videos.shape[0]}" 
    #     )
    #     assert item.video_grid_thw.shape[0] == item.video_second_per_grid.shape[0], (
    #         f"data error: video_grid_thw.shape[0]={item.video_grid_thw.shape[0]}, "
    #         f"video_second_per_grid[0]={item.video_second_per_grid.shape[0]}"
    #     )
    #     print(f'idx: {idx}')
    #     for key, value in asdict(item).items():
    #         if isinstance(value, list):
    #             print(f"{key}: {len(value)}")
    #         elif isinstance(value, torch.Tensor):
    #             print(f'{key}: {value.shape}')
            
