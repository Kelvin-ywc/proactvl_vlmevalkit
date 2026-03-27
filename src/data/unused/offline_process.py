import os
import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor
from src.data.custom_commentary_dataset import SupervisedStreamDatasetSample, DataCollatorForStream2Text, construct_conversation_prompt
from src.utils.proact_process import get_audio_chunks_from_one_video, process_interleave_mm_info, process_interleave_vision_info
from torch.utils.data import Dataset
import json

class CustomCommentaryDataset(Dataset):
    def __init__(self, video_dir_path: str, ann_path: str, processor, use_audio_in_video, is_train=True, video_clip_length=90, commentary_drop_rate=0.0, data_percentage=1.0, cache_dir=None):
        super().__init__()
        self.processor = processor
        self.video_dir_path = video_dir_path
        self.ann_path = ann_path
        self.use_audio_in_video = use_audio_in_video
        self.data_type = torch.bfloat16

        self.ann_list = []
        self.conversation_info_list = []

        self.commentary_drop_rate = commentary_drop_rate
        self.data_percentage = data_percentage
        self.cache_dir = cache_dir

        with open(self.ann_path, 'r') as f:
            for line in f:
                if line.strip():
                    ann = json.loads(line)
                    self.ann_list.append(ann)
        # FIXME hard code split
        ann_len = len(self.ann_list)
        if is_train:
            self.ann_list = self.ann_list[:int(ann_len*0.9)]
        else:
            self.ann_list = self.ann_list[int(ann_len*0.9):]
        if data_percentage < 1.0:
            self.ann_list = self.ann_list[:int(len(self.ann_list)*data_percentage)]
        print(f'Load {len(self.ann_list)} samples from {self.ann_path}')

        # construct conversations
        for ann in self.ann_list:
            conversation_info = {
                'ann': ann,
                'video_dir_path': self.video_dir_path,
            }
            self.conversation_info_list.append(conversation_info)
    
    def __len__(self):
        return len(self.conversation_info_list)

    def __getitem__(self, index):
        conversation_info = self.conversation_info_list[index]
        cur_conversation = construct_conversation_prompt(conversation_info['ann'], conversation_info['video_dir_path'])
        text = self.processor.apply_chat_template(cur_conversation, add_generation_prompt=False, tokenize=False)

        video_path = conversation_info['ann']['video_path']
        start_time = conversation_info['ann']['video_begin']
        end_time = conversation_info['ann']['video_end']
        cache_file_name = f"{os.path.basename(video_path)}_{start_time}_{end_time}.pt"
        rel_dir = os.path.dirname(video_path)
        base_name = os.path.basename(video_path)
        cache_file_name = f'{base_name}_{start_time}_{end_time}.pt'
        cache_file_path = os.path.join(self.cache_dir, rel_dir, cache_file_name)

        audios, images, videos = process_interleave_mm_info(cur_conversation, self.use_audio_in_video, return_video_kwargs=False)
        
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt",
            padding=True, 
            use_audio_in_video=self.use_audio_in_video,
            min_pixels=128*28*28,
            max_pixels=128*28*28,
        )

        # save to cache
        for k in list(inputs.keys()):
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                inputs[k] = _to_half_if_float(v.cpu())  # 一律保存 CPU + fp16
        torch.save(inputs, cache_file_path)
        del audios, images, videos
        return SupervisedStreamDatasetSample(
            **inputs,
            input_features=None,
            feature_attention_mask=None,
            labels=None,
            active_labels=None,
        )


def _to_half_if_float(t):
    return t.half() if isinstance(t, torch.Tensor) and t.dtype == torch.float32 else t


if __name__ == "__main__":
    processor = Qwen2_5OmniProcessor.from_pretrained(
        'Qwen/Qwen2.5-Omni-7B'
    )

    video_dir_path = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos'
    ann_path = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/processed_annotations/lol_merged_split_30s_overlap15s_filtered.jsonl'
    cache_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    dataset = CustomCommentaryDataset(
        video_dir_path=video_dir_path,
        ann_path=ann_path,
        processor=processor,
        use_audio_in_video=False,
        is_train=True,
        video_clip_length=30,
        commentary_drop_rate=0.0,
        data_percentage=1.0,
        cache_dir=cache_dir,
    )

    # dataset = CustomCommentaryDataset(
    #     video_dir_path=video_dir_path,
    #     ann_path=ann_path,
    #     processor=processor,
    #     use_audio_in_video=False,
    #     is_train=False,
    #     video_clip_length=30,
    #     commentary_drop_rate=0.0,
    #     data_percentage=1.0,
    #     cache_dir=None,
    # )
    collator = DataCollatorForStream2Text(processor.tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collator, num_workers=4)
    idx = 0
    for batch in tqdm(dataloader):
        print(f'Process batch {idx}')
        exit()