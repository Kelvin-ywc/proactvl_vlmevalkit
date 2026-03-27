import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import debugpy
import torch
# from datasets import load_dataset, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from constants import (ACTIVE_GEN_TOKEN, DEFAULT_ASSISTANT_ROLE_TOKEN,
                       DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                       DEFAULT_SYSTEM_ROLE_TOKEN, DEFAULT_USER_ROLE_TOKEN,
                       IGNORE_INDEX, SILENT_TOKEN)
from conversations import S2T_SYSTEM_PROMPT, S2T_SYSTEM_PROMPT_PROACTIVE
from transformers import Qwen2_5OmniProcessor
from utils.qwen_omni_utils import (extract_vision_info, fetch_image,
                                   fetch_video, process_audio_info,
                                   process_mm_info, process_stream_info, process_vision_info,
                                   smart_resize)

# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger to connect...")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Error starting debugpy: {e}")
    
    
@dataclass
class SupervisedDatasetSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    feature_attention_mask: torch.Tensor
    input_features: torch.Tensor
    # audio_segment_pos: torch.Tensor

@dataclass
class SupervisedStreamDatasetSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    video_grid_thw: torch.Tensor
    pixel_values_videos: torch.Tensor
    conversations: list
# Support Stream Video without Audio
class Stream2TextSupervisedDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 processor: Qwen2_5OmniProcessor,
                 use_audio_in_video=False,
                 ):
        super(Stream2TextSupervisedDataset, self).__init__()
        
        self.processor = processor

        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        # print(self.raw_data[1])
    
    def __len__(self) -> int:
        return len(self.raw_data)
    
    def prepare_labels_for_multimodal(self,
                                    input_ids: torch.Tensor,) -> torch.Tensor:
        labels = input_ids.clone()
        # mask audio token <|AUDIO|>
        # audio_token_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_token))
        # print(audio_token_mask)
        # labels = labels.masked_fill(audio_token_mask, IGNORE_INDEX)
        # mask system prompt
        im_start_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN))[1]
        im_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN))[1]
        silent_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(SILENT_TOKEN))
        active_gen_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(ACTIVE_GEN_TOKEN))
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
        
        return labels
    def __getitem__(self, index: int):
        data_item = self.raw_data[index]
        base_video_dir = "/home/v-yuhongdai/AI-Commentary/dataset/process_video_ffmpeg/concat"
        video_path = os.path.join(base_video_dir, data_item[0]['metadata']['video_id'])
        print(data_item[0])
        conversation = data_item[0]['conversation']
        # print(video_path)
        # print(conversation)
        # print(data_item[0]['conversation'])
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        # print(f"Processing item {index}: {text}")
        # if source == 'VoiceAssistant':
        audios, images, videos = process_stream_info(conversations=data_item, video_path=video_path, return_video_kwargs=False)
        
        # print(text)
        inputs = self.processor(text=text, 
                                audios=audios, 
                                images=images, 
                                videos=videos, 
                                return_tensors="pt", 
                                padding=True, 
                                use_audio_in_video=False,
                                add_gen_token=False,
                                add_silent_token=False)
        # print(inputs)
        # print(self.processor.tokenizer)
        input_ids = inputs['input_ids']
        labels = self.prepare_labels_for_multimodal(input_ids)
        # print(inputs)
        # print(inputs)
        # data_dict = (
        return SupervisedStreamDatasetSample(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            labels=labels,
            pixel_values_videos=inputs['pixel_values_videos'],
            video_grid_thw=inputs['video_grid_thw'],
            conversations=conversation,
            # audio_segment_pos=inputs['audio_segment_pos']
        )
        
class Audio2TextSupervisedDataset(Dataset):
    def __init__(self,
                 data_path,
                 processor,
                 add_gen_token=True,
                 add_silent_token=True,
                 use_audio_in_video=False,
                 use_default_system_prompt=False,
                 ignore_assistant_ans=False):
        super(Audio2TextSupervisedDataset, self).__init__()
        
        self.processor = processor
        # self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        self.use_audio_in_video = use_audio_in_video
        self.add_gen_token = add_gen_token
        self.add_silent_token = add_silent_token
        self.use_default_system_prompt = use_default_system_prompt
        self.ignore_assistant_ans = ignore_assistant_ans
        
        if self.add_gen_token:
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [ACTIVE_GEN_TOKEN]})
        if self.add_silent_token:
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [SILENT_TOKEN]})
            
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        # if os.path.exists(data_path):
        #     self.raw_data = load_from_disk(data_path)
        # else:
        #     self.raw_data = load_dataset(data_path)['train']
        
    
    def prepare_labels_for_multimodal(self,
                                      input_ids: torch.Tensor,) -> torch.Tensor:
        labels = input_ids.clone()
        # mask audio token <|AUDIO|>
        audio_token_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_token))
        # print(audio_token_mask)
        # labels = labels.masked_fill(audio_token_mask, IGNORE_INDEX)
        # mask system prompt
        im_start_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN))[1]
        im_end_index = torch.where(labels == self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN))[1]
        silent_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(SILENT_TOKEN))
        active_gen_mask = (labels == self.processor.tokenizer.convert_tokens_to_ids(ACTIVE_GEN_TOKEN))
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
                    if self.ignore_assistant_ans:
                        labels[0][im_start_idx:im_end_idx + 2] = IGNORE_INDEX
                    else:
                        labels[0][im_start_idx:im_start_idx + 3] = IGNORE_INDEX
        
        labels.masked_fill_(active_gen_mask, self.processor.tokenizer.convert_tokens_to_ids(ACTIVE_GEN_TOKEN))
        labels.masked_fill_(silent_mask, self.processor.tokenizer.convert_tokens_to_ids(SILENT_TOKEN))
        
        return labels
    
    def __len__(self) -> int:
        return len(self.raw_data)
        # self.data = load_dataset()
    
    def __getitem__(self,
                    index: int):
        data_item = self.raw_data[index]
        # print(data_item)
        source = data_item.get('source', None)
        conv = data_item['conversation']
        # print(conv)
        if self.use_default_system_prompt:
            conv.insert(0, S2T_SYSTEM_PROMPT)
        else:
            conv.insert(0, S2T_SYSTEM_PROMPT_PROACTIVE) 
        
        text = self.processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)

        if source == 'VoiceAssistant':
            audios, images, videos = process_mm_info(conv, use_audio_in_video=self.use_audio_in_video)
        
        # print(text)
        inputs = self.processor(text=text, 
                                audios=audios, 
                                images=images, 
                                videos=videos, 
                                return_tensors="pt", 
                                padding=True, 
                                use_audio_in_video=self.use_audio_in_video,
                                add_gen_token=self.add_gen_token,
                                add_silent_token=self.add_silent_token)
        # print(inputs)
        # print(self.processor.tokenizer)
        input_ids = inputs['input_ids']
        labels = self.prepare_labels_for_multimodal(input_ids)
        # print(inputs)
        # data_dict = (
        return SupervisedDatasetSample(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            labels=labels,
            feature_attention_mask=inputs['feature_attention_mask'],
            input_features=inputs['input_features'],
            # audio_segment_pos=inputs['audio_segment_pos']
        )
        # data_dict = dict()
        # return 
        

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
        _batch_pixel_values_videos_mask = torch.stack(pixel_values_videos_list, dim=0)
        # _batch_video_grid_thw = torch.stack(video_grid_thw_list, dim=0)
        _batch_video_grid_thw = torch.cat(video_grid_thw_list, dim=0)
        _batch_video_second_per_grid = torch.cat(video_second_per_grid_list, dim=0)
        _batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        _batch_active_labels = pad_sequence(active_labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        # return None
        return {
            'input_ids': _batch_input_ids,
            'attention_mask': _batch_attention_mask,
            'pixel_values_videos': _batch_pixel_values_videos_mask,
            'video_grid_thw': _batch_video_grid_thw,
            'video_second_per_grid': _batch_video_second_per_grid,
            'labels': _batch_labels,
            'active_labels': _batch_active_labels,
            "conversations": _batch_conversations,
        }

class DataCollatorForAudio2Text(object):
    """Collate examples for Audio2Text supervised fine-tuning."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, samples):

        input_ids_list = [sample.input_ids.squeeze(0) for sample in samples]
        attention_mask_list = [sample.attention_mask.squeeze(0) for sample in samples]
        labels_list = [sample.labels.squeeze(0) for sample in samples]
        feature_attention_mask_list = [sample.feature_attention_mask.squeeze(0) for sample in samples]
        input_features_list = [sample.input_features.squeeze(0) for sample in samples]
        # audio_segment_pos_list = [sample.audio_segment_pos.squeeze(0) for sample in samples]
        
        _batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
        _batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        _batch_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        _batch_feature_attention_mask = torch.stack(feature_attention_mask_list, dim=0)
        _batch_input_features = torch.stack(input_features_list, dim=0)
        
        return {
            'input_ids': _batch_input_ids,
            'attention_mask': _batch_attention_mask,
            'labels': _batch_labels,
            'feature_attention_mask': _batch_feature_attention_mask,
            'input_features': _batch_input_features,
        }
    

if __name__ == "__main__":
    # Example usage
    import argparse
    parse = argparse.ArgumentParser(description="AudioOnlySupervisedDataset")
    parse.add_argument("--debug", action="store_true", help="Debug mode")
    args = parse.parse_args()
    if args.debug:
        import debugpy
        try:
            debugpy.listen(("localhost", 9501))
            debugpy.wait_for_client()
        except Exception as e:
            print(f"Error starting debugpy: {e}")
            
    data_path = Path('dataset/hf_dataset')
    # data_path = './dataset/VoiceAssistant_one-turn_data_5K.json'
    data_path = "/home/aiscuser/AI-Commentary/dataset/refresh_commentary/all_sec_sft_json/json_dataset.json"
    # dataset = Audio2TextSupervisedDataset(data_path)
    dataset = Stream2TextSupervisedDataset(data_path, processor=Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B"))
    # input_ids = torch.tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,    264,
    #        4108,   3738,   7881,    553,    279,   1207,  16948,   7909,     11,
    #       54364,   5737,     11,  12875,    315,    817,  46344,  82529,    323,
    #        9124,  11127,     11,    438,   1632,    438,  23163,   1467,    323,
    #        8806,     13,   1446,    646,   5530,    979,    498,   1184,    311,
    #         462,  63019,   6923,  14507,    311,   1492,   3847,     13, 151645,
    #         198, 151644,    872,    198, 151647, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646,
    #      151646, 151646, 151646, 151646, 151648, 151665, 151645,    198, 151644,
    #       77091,    198,   9707,      0,    358,   2776,  85225,     11,    458,
    #       15235,   7743,  17847,   6188,    311,   7789,    498,    448,   4755,
    #          11,   9462,     11,    323,   9115,     13,   3017,  16928,   2525,
    #         504,  10847,   4862,    389,   7988,  70403,     11,  27362,    752,
    #         311,   3410,   1931,   7246,   7699,   5980,   9904,     13,    358,
    #        2776,   1588,    311,   1492,    498,    448,    264,   8045,    315,
    #        9079,    323,   1281,    697,   2272,   8661,      0, 151645,    198]])
    # print(len(dataset))
    dataset[1]
    # print(dataset.prepare_labels_for_multimodal(input_ids))