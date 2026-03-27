from src.data.custom_commentary_dataset import CustomCommentaryDataset, DataCollatorForStream2Text
from transformers import Qwen2_5OmniProcessor
import torch
from tqdm import tqdm


ANN_PATH = {
    'lol': '/home/v-weicaiyan/ds/DATA/ann/all_in_one/lol_split_clips_60s_overlap40s_combined.jsonl',
    'soccer': '/home/v-weicaiyan/ds/DATA/ann/all_in_one/soccer_split_clips_60s_overlap40s_combined.jsonl',
    'livecc': '/home/v-weicaiyan/ds/DATA/ann/all_in_one/livecc_split_clips_60s_combined.jsonl',
    'minecraft': '/home/v-weicaiyan/ds/DATA/ann/all_in_one/minecraft_split_clips_60s_overlap40s_combined.jsonl',
    'csgo': '/home/v-weicaiyan/ds/DATA/ann/all_in_one/csgo_split_clips_60s_overlap40s_combined.jsonl',
}
DATA_NAME_TO_VIDEO = {
    'lol': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos',
    'soccer': '/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet',
    'livecc': '/home/v-weicaiyan/ds/DATA/live_sft/videos/videos',
    'csgo': '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/videos',
    'minecraft': "/home/v-weicaiyan/ds/DATA/game_commentary/minecraft/videos"
}

if __name__ == '__main__':
    processor = Qwen2_5OmniProcessor.from_pretrained(
        'Qwen/Qwen2.5-Omni-7B'
    )
    dataset_name = 'lol'
    dataset = CustomCommentaryDataset(
        dataset_names=[dataset_name],
        video_dir_path=[DATA_NAME_TO_VIDEO[dataset_name]],
        ann_path=[ANN_PATH[dataset_name]],
        processor=processor,
        use_audio_in_video=False
    )
    cnt00 = 0
    cnt01 = 0
    cnt10 = 0
    cnt11 = 0
    cnt = 0
    collator = DataCollatorForStream2Text(processor.tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=DataCollatorForStream2Text(processor.tokenizer), shuffle=True, num_workers=8)
    for idx, sample in tqdm(enumerate(dataloader)):
        # print(sample)
        active_labels = sample['active_labels']
        # 遍历，统计00，01，10，11的情况
        for active_label in active_labels:
            active_label = active_label[torch.where(active_label!=-100)]
            for i in range(1, len(active_label)):
                pre = active_label[i-1]
                cur = active_label[i]
                if pre==0 and cur==0:
                    cnt00 += 1
                elif pre == 0 and cur == 1:
                    cnt01 += 1
                elif pre ==1 and cur == 0:
                    cnt10 += 1
                elif pre==1 and cur == 1:
                    cnt11 += 1
                cnt += 1 
        print(f'00: {cnt00}, 01: {cnt01}, 10: {cnt10}, 11: {cnt11}')
    print(f'00: {cnt00}, 01: {cnt01}, 10: {cnt10}, 11: {cnt11}')
