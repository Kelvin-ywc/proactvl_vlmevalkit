'''
This file is used to split the custom annotation into clips for training.

Support dataset types:
1. soccernet 原始数据为长视频，需要进行切片，同时补充history信息
2. lol 原始数据为长视频，需要进行切片，同时补充history信息，同时，lol为多人解说数据，需要分角色，对于未激活的解说员，解说词要推后一秒。
3. minecraft 格式同livecc，需要进行切片。
4. livecc(通用) livecc一条数据即为一条训练样本，根据clip_duration过滤掉太长的样本

训练标签中一共有4个角色
1. system
2. user
3. background
4. assistant

数据集：
1. soccernet
2. livecc
3. lol
4. minecraft

数据流
merge all json files -> [standard format jsonl: standard_format.jsonl] 
-> split speakers -> [split_speaker.jsonl] 
-> split into seconds -> [split_seconds.jsonl] 
-> split into clips -> [split_clips_xxs_overlapxxs.jsonl] 
'''
from pathlib import Path
import decord
import math
import os
import json
import ffmpeg

ANN_PATH = {
    'lol': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/lol_standard_format.jsonl',
    'soccernet': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/soccer_standard_format.jsonl',
    'livecc': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/livecc_standard_format.jsonl',
}

def split_lol_annotations_into_clips(ann_path, save_path, clip_duration, overlap):
    pass
def split_livecc_annotations_into_clips(ann_path, save_path, min_clip_duration, max_clip_duration):
    with open(ann_path, 'r') as f_in, open(save_path, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            # print(ann)
            # ann['speaker'] = 'SPEAKER_0'
            # ann['active'] = True
            offset = ann['video_begin']
            new_commentary = []
            new_commentary.append({
                'query': ann['annotations'][0]['query'],
                'begin_time': 0,
                'end_time': 1,
                'role': 'user',
            })
            if ann['duration'] <= max_clip_duration:

                ann['video_begin'] = 0
                ann['video_end'] = ann['duration']
                # livecc仅有一条数据
                ann['annotations'][0]['begin_time'] -= offset
                ann['annotations'][0]['end_time'] -= offset
                for commentary in ann['annotations'][0]['commentary']:
                    commentary['begin_time'] -= offset
                    commentary['end_time'] -= offset
                    commentary['speaker'] = 'SPEAKER_0'
                    commentary['active'] = True
                    commentary['role'] = 'assistant'
                    new_commentary.append(commentary)
                ann['annotations'] = new_commentary
                f_out.write(json.dumps(ann, ensure_ascii=False) + "\n")
            elif ann['duration'] < min_clip_duration:
                continue
            else:
                ann['video_begin'] = 0
                ann['video_end'] = max_clip_duration
                ann['duration'] = max_clip_duration
                ann['annotations'][0]['begin_time'] -= offset
                ann['annotations'][0]['end_time'] = max_clip_duration
                new_commentary = []
                for commentary in ann['annotations'][0]['commentary']:
                    commentary['begin_time'] -= offset
                    commentary['end_time'] -= offset
                    commentary['speaker'] = 'SPEAKER_0'
                    commentary['active'] = True
                    commentary['role'] = 'assistant'
                    if commentary['end_time'] <= max_clip_duration:
                        new_commentary.append(commentary)
                    else:
                        break
                ann['annotations'] = new_commentary
                f_out.write(json.dumps(ann, ensure_ascii=False) + "\n")
            # print(ann)


def split_soccernet_annotations_into_clips(ann_path, save_path, min_clip_duration, max_clip_duration, overlap):
    with open(ann_path, 'r') as f_in, open(save_path, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            print(ann)
            speakers = ann['speakers']
            new_ann = {
                'video_path': ann['video_path'],
            }
            for speaker in speakers:
                pass
            return None

def split_annotations_into_clips(dataset_types, min_clip_duration, max_clip_duration, overlap):
    for dataset_type in dataset_types:
        ann_path = ANN_PATH[dataset_type]
        save_path = ann_path.replace('standard_format', f'split_clips_{max_clip_duration}s_overlap{overlap}s')
        if dataset_type == 'lol':
            split_lol_annotations_into_clips(ann_path, save_path, min_clip_duration, max_clip_duration, overlap)
        elif dataset_type == 'soccernet':
            split_soccernet_annotations_into_clips(ann_path, save_path, min_clip_duration, max_clip_duration, overlap)
        elif dataset_type == 'livecc':
            split_livecc_annotations_into_clips(ann_path, save_path, min_clip_duration, max_clip_duration)
        else:
            raise NotImplementedError(f"Dataset type {dataset_type} not supported.")

if __name__ == "__main__":
    min_clip_duration = 30  # seconds
    max_clip_duration = 60
    overlap = 20

    split_annotations_into_clips(['livecc'], min_clip_duration, max_clip_duration, overlap)
    # split_annotations_into_clips(['soccernet'], min_clip_duration, max_clip_duration, overlap)