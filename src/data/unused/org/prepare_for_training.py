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

You are SPEAKER_0
<system> # 系统提示
<background> # 会话背景信息
<user:video> # 第一秒画面
<assistant:SPEAKER_0> # assistant进行解说
<background:SPEAKER_1> # 获取别的解说员解说
<user:video|query> # 第二秒画面，同时用户有新的提问
<assistant:SPEAKER_0> # assistant中止解说，对用户提问进行回答

最终用于训练的标签格式如下：
{
    "video_path": "...",
    "video_begin": 0,
    "video_end": 60,
    "duration": 60,
    "speakers": ["SPEAKER_0", "SPEAKER_1"],
    "annotations": [
        {
            "history": "...",
            "query": "...",
            "begin_time": 0,
            "end_time": 1,
            "role": "user",
        },
        {
            "begin_time": 1,
            "end_time": 2,
            "speaker": "SPEAKER_0",
            "role": "assistant",
            "commentary": "..."
        },
        {
            "history": '...', 
            "query": "...",
            "begin_time": 3,
            "end_time": 4,
            "role": "user",
        }
    ]
}
在会话开始首先需要构建history，需要记录之前会话的内容，包括别的解说员的内容，自己的解说内容，用户提问
user有三种类型，一个是用户提问带query，一个是之前其他解说员的commentary，放在history里，一个是带history，仅放在会话开始

90s 60s
0-90 30-120 60-150
[0, 1, 2] -> [1, 2, 3]

<system>
<user: [history][V]<vision_eos>Please comment ...>
<assistant>

livecc和minecraft数据不需要根据active rate过滤，soccer和lol数据需要根据active rate过滤
'''
from pathlib import Path
import decord
import math
import os
import json
import ffmpeg
# from src.utils.utils import split_text_into_segments
# ANN_PATH = {
#     'lol': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/lol_standard_format.jsonl',
#     'soccer': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/soccer_standard_format.jsonl',
#     'livecc': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/livecc_standard_format.jsonl',
#     'minecraft': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/minecraft_split_speaker.jsonl',
#     'csgo': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/csgo_standard_format.jsonl',
# }
ANN_PATH = {
    'lol': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_standard_format.jsonl',
    'csgo': '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_standard_format.jsonl',
    'black_myth_wukong': '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_standard_format.jsonl',
    'cyberpunk': '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_standard_format.jsonl',
    'livecc': '/home/v-weicaiyan/ds/DATA/live_sft/ann/livecc_standard_format.jsonl',
    'livecc_v2': '/home/v-weicaiyan/ds/DATA/live_sft_6k_12k/ann/livecc_v2_standard_format.jsonl',
    'livecc_v3': '/home/v-weicaiyan/ds/DATA/live_sft_12k_24k/ann/livecc_v3_standard_format.jsonl',
    'soccer': '/home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_standard_format.jsonl',
    'minecraft': '/home/v-weicaiyan/ds/DATA/game_commentary/minecraft/ann/minecraft_split_speaker.jsonl',
}

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
    return segments

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
                'history': ann['history'],
                'query': ann['annotations'][0]['query'],
                'begin_time': 0,
                'end_time': 1,
                'role': 'user',
            })
            # ann删除history，改为放在第一条commentary中
            del ann['history']
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
                    commentary['role'] = 'assistant'
                    new_commentary.append(commentary)
                ann['annotations'] = new_commentary
                f_out.write(json.dumps(ann, ensure_ascii=False) + "\n")
            elif ann['duration'] < min_clip_duration:
                continue
            else:
                # continue
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
                    commentary['role'] = 'assistant'
                    if commentary['end_time'] <= max_clip_duration:
                        new_commentary.append(commentary)
                    else:
                        break
                ann['annotations'] = new_commentary
                f_out.write(json.dumps(ann, ensure_ascii=False) + "\n")
            # print(ann)

def split_speakers(ann_path, save_path):
    idx = 0
    with open(ann_path, 'r') as f_in, open(save_path, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            print(ann)
            speakers = ann['speakers']
            for speaker in speakers:
                idx += 1
                new_ann = {
                    'video_path': ann['video_path'],
                    'video_begin': ann['video_begin'],
                    'video_end': ann['video_end'],
                    'duration': ann['duration'],
                    'speakers': ann['speakers'],
                    'active_speaker': speaker,
                    'annotations': []
                }
                for annotation in ann['annotations']:
                    if annotation['speaker'] == speaker:
                        # active speaker设置为assistant，非active speaker设置为user
                        new_annotation = {
                            'role': 'assistant',
                            'begin_time': annotation['begin_time'],
                            'end_time': annotation['end_time'],
                            'speaker': annotation['speaker'],
                            'commentary': annotation['commentary']
                        }
                    else:
                        new_annotation = {
                            'role': 'user',
                            'begin_time': annotation['begin_time'],
                            'end_time': annotation['end_time'],
                            'query': annotation['query'],
                            'speaker': annotation['speaker'],
                            'commentary': annotation['commentary']
                        }
                    new_ann['annotations'].append(new_annotation)
                f_out.write(json.dumps(new_ann, ensure_ascii=False) + "\n")
    print(f"Total split into {idx} speaker-specific annotations.")


def sentence_to_seconds(item):
    new_annotations = []
    print(item)
    if item['role'] == 'assistant':

        text_chunks = split_text_into_segments(item['commentary'], item['end_time'] - item['begin_time'])

        for i, chunk in enumerate(text_chunks):
            new_annotations.append({
                'role': 'assistant',
                'begin_time': item['begin_time'] + i,
                'end_time': item['begin_time'] + i + 1,
                'commentary': ' '+chunk,
                'speaker': item['speaker']
            })
    elif item['role'] == 'user':
        first_ann = None
        if item['query'] != '':
            new_annotations.append({
                'role': 'user',
                'begin_time': item['begin_time'],
                'end_time': item['begin_time'] + 1,
                'query': item['query'],
                'speaker': 'user'
            })
        if 'commentary' in item and item['commentary'] != '':
            text_chunks = split_text_into_segments(item['commentary'], item['end_time'] - item['begin_time'])
            for i, chunk in enumerate(text_chunks):
                # if item['begin_time'] + i + 2 <= item['end_time']:
                new_annotations.append({
                    'role': 'user',
                    'begin_time': item['begin_time'] + i + 1,
                    'end_time': item['begin_time'] + i + 2,
                    'commentary': ' '+chunk,
                    'speaker': item['speaker']
                })
    print(new_annotations)
    return new_annotations

# 过滤长度为0秒的样本
def split_into_seconds(ann_path, save_path):
    with open(ann_path, 'r') as f_in, open(save_path, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            # print(ann)
            new_annotations = []
            for item in ann['annotations']:
                if item['end_time'] - item['begin_time'] <=0:
                    continue
                if 'commentary' in item and item['commentary'] == {}:
                    continue
                new_annotations.extend(sentence_to_seconds(item))
            ann['annotations'] = new_annotations
            f_out.write(json.dumps(ann, ensure_ascii=False) + "\n")

# 1.补充clip前history，将其他解说员的commentary转化为history
def extract_ann_clip(ann, history_begin, clip_video_begin, clip_video_end):
    # print(ann)
    print(f"history_begin: {history_begin}, clip_video_begin: {clip_video_begin}, clip_video_end: {clip_video_end}")

    res = []
    new_ann = {
        'video_path': ann['video_path'],
        'video_begin': clip_video_begin,
        'video_end': clip_video_end,
        'duration': clip_video_end - clip_video_begin,
        # 'speakers': ann['speakers'],
        'annotations': []
    }
    # 提取history
    history_texts = []
    for item in ann['annotations']:
        if item['end_time'] <= clip_video_begin and item['begin_time'] >= history_begin:
            if item['role'] == 'assistant':
                history_texts.append(f"(YOU):{item['commentary']}\n")
            elif item['role'] == 'user':
                if 'query' in item and item['query'] != '':
                    history_texts.append(f"(USER):{item['query']}\n")
                elif 'commentary' in item:
                    history_texts.append(f"({item['speaker']}):{item['commentary']}\n")
                else:
                    raise ValueError(f"Unknown user annotation format: {item}")
            else:
                raise ValueError(f"Unknown role in annotation: {item}")
    history = ''.join(history_texts).strip()
    if history != '':
        new_ann['annotations'].append({
            'role': 'user',
            'begin_time': clip_video_begin,
            'end_time': clip_video_begin + 1,
            'history': history
        })
    # 提取clip内的annotation
    for item in ann['annotations']:
        if item['begin_time'] >= clip_video_begin and item['end_time'] <= clip_video_end:
            if item['role'] == 'assistant':
                new_ann['annotations'].append(item)
            elif item['role'] == 'user' and 'query' in item and item['query'] != '':
                new_ann['annotations'].append(item)
            elif item['role'] == 'user' and 'commentary' in item:
                new_ann['annotations'].append({
                    'role': 'user',
                    'begin_time': item['begin_time'],
                    'end_time': item['end_time'],
                    'history': f'({item["speaker"]}):{item["commentary"]}\n'
                })
            else:
                print(f"[warn] Unknown user annotation format in clip: {item}")
                raise ValueError(f"Unknown user annotation format in clip: {item}")
    if len(new_ann['annotations']) > 0:
        return new_ann
    return None

def split_into_clips(ann_path, save_path, min_clip_duration, clip_duration, move_duration, min_active_rate=0.0, max_active_rate=1.0, filter_silence=False):
    with open(ann_path, 'r') as f_in, open(save_path, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            # 对ann进行裁剪，duration 为clip_duration，重叠overlap，提取前overlap作为history
            history_begin = -move_duration
            clip_video_begin = 0
            clip_video_end = clip_duration
            while clip_video_begin < ann['duration']:

                if clip_video_end - clip_video_begin < min_clip_duration:
                    break
                if clip_video_end - clip_video_begin < min_clip_duration:
                    break
                res = extract_ann_clip(ann, history_begin, clip_video_begin, clip_video_end)
                if res is not None:
                    res['video_begin'] = clip_video_begin
                    res['video_end'] = clip_video_end
                    res['duration'] = clip_video_end - clip_video_begin
                    # 过滤静音片段
                    if filter_silence:
                        # 统计active比例
                        total_duration = res['duration']
                        active_duration = 0
                        for item in res['annotations']:
                            if item['role'] == 'assistant' and 'commentary' in item and item['commentary'].strip() != '':
                                active_duration += 1
                        active_rate = active_duration / total_duration
                        # min_active_rate <rate<= max_active_rate
                        if active_rate <= min_active_rate or active_rate > max_active_rate:
                            history_begin += move_duration
                            clip_video_begin += move_duration
                            clip_video_end += move_duration
                            if clip_video_end > ann['duration']:
                                clip_video_end = ann['duration']
                            continue
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

                history_begin += move_duration
                clip_video_begin += move_duration
                clip_video_end += move_duration
                if clip_video_end > ann['duration']:
                    clip_video_end = ann['duration']

def split_multi_speakers_train_eval_test(ann_path, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1):
    assert math.isclose(train_ratio + eval_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    base_path = str(Path(ann_path).parent)
    train_path = str(Path(ann_path)).replace('split_speaker', 'train_split_speaker')
    eval_path = str(Path(ann_path)).replace('split_speaker', 'eval_split_speaker')
    test_path = str(Path(ann_path)).replace('split_speaker', 'test_split_speaker')


    with open(ann_path, 'r') as f_in:
        lines = f_in.readlines()
        total_lines = len(lines)
        eval_len = int(total_lines * eval_ratio)
        test_len = int(total_lines * test_ratio)
        train_len = total_lines - eval_len - test_len
        

        with open(train_path, 'w') as f_train:
            for line in lines[:train_len]:
                f_train.write(line)

        with open(eval_path, 'w') as f_eval:
            for line in lines[train_len:train_len + eval_len]:
                f_eval.write(line)

        with open(test_path, 'w') as f_test:
            for line in lines[train_len + eval_len:train_len + eval_len + test_len]:
                f_test.write(line)
    return train_path, eval_path, test_path  

def split_train_val_test(ann_path, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1):
    assert math.isclose(train_ratio + eval_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    base_path = str(Path(ann_path).parent)
    train_path = str(Path(ann_path)).replace('standard_format', 'train_standard_format')
    eval_path = str(Path(ann_path)).replace('standard_format', 'eval_standard_format')
    test_path = str(Path(ann_path)).replace('standard_format', 'test_standard_format')


    with open(ann_path, 'r') as f_in:
        lines = f_in.readlines()
        total_lines = len(lines)
        eval_len = int(total_lines * eval_ratio)
        test_len = int(total_lines * test_ratio)
        train_len = total_lines - eval_len - test_len
        

        with open(train_path, 'w') as f_train:
            for line in lines[:train_len]:
                f_train.write(line)

        with open(eval_path, 'w') as f_eval:
            for line in lines[train_len:train_len + eval_len]:
                f_eval.write(line)

        with open(test_path, 'w') as f_test:
            for line in lines[train_len + eval_len:train_len + eval_len + test_len]:
                f_test.write(line)
    return train_path, eval_path, test_path  

def split_annotations_into_clips(dataset_types, min_clip_duration, clip_duration, overlap, min_active_rate=0.0, max_active_rate=1.0):
    min_active_rate_for_file_name = str(int(min_active_rate * 100))
    max_active_rate_for_file_name = str(int(max_active_rate * 100))
    for dataset_type in dataset_types:
        ann_path = ANN_PATH[dataset_type]
        # livecc 特殊处理，因为其已经提供好了history内容，不需要再补充history，同时一样livecc数据对应一个训练样本，只需要过滤掉过长和过短的样本即可
        if 'livecc' in dataset_type:
            train_path, eval_path, test_path = split_train_val_test(ann_path)
            for ann_path in [train_path, eval_path]:
                save_clips_path = ann_path.replace('standard_format', f'split_clips_{clip_duration}s')
                split_livecc_annotations_into_clips(ann_path, save_clips_path, min_clip_duration, clip_duration)
            continue
        if dataset_type in ['minecraft']:
            # minecraft数据已经是split_speaker格式，跳过split_speaker步骤
            save_speaker_path = ann_path
            train_split_speaker_path, eval_split_speaker_path, test_split_speaker_path = split_multi_speakers_train_eval_test(save_speaker_path)
            for ann_path in [train_split_speaker_path, eval_split_speaker_path]:
                # 2. split into seconds
                save_speaker_path = ann_path
                save_seconds_path = save_speaker_path.replace('split_speaker', 'split_seconds')
                split_into_seconds(save_speaker_path, save_seconds_path)
                
                save_clips_path = save_seconds_path.replace('split_seconds', f'split_clips_{clip_duration}s_overlap{overlap}s')
                split_into_clips(save_seconds_path, save_clips_path, min_clip_duration, clip_duration, clip_duration-overlap,filter_silence=True)
            continue
        # 0. split train, val, test, for [lol,csgo,black_myth_wukong,cyberpunk,soccer]
        train_path, eval_path, test_path = split_train_val_test(ann_path)
        for ann_path in [train_path, eval_path]:
            # 1. split speakers
            save_speaker_path = ann_path.replace('standard_format', 'split_speaker')
            split_speakers(ann_path, save_speaker_path)
            # 2. split into seconds
            save_seconds_path = save_speaker_path.replace('split_speaker', 'split_seconds')
            split_into_seconds(save_speaker_path, save_seconds_path)
            # 3. split into clips
            save_clips_path = save_seconds_path.replace('split_seconds', f'split_clips_{clip_duration}s_overlap{overlap}s_active{min_active_rate_for_file_name}-{max_active_rate_for_file_name}')
            split_into_clips(save_seconds_path, save_clips_path, min_clip_duration, clip_duration, clip_duration-overlap, min_active_rate=min_active_rate,max_active_rate=max_active_rate,filter_silence=True)

if __name__ == "__main__":
    min_clip_duration = 12  # seconds
    clip_duration = 36
    overlap = 18
    min_active_rates = [0.0, 0.3, 0.7]
    max_active_rates = [0.3, 0.7, 1.0]
    for min_active_rate, max_active_rate in zip(min_active_rates, max_active_rates):
        # split_annotations_into_clips(['lol'], min_clip_duration, clip_duration, overlap, min_active_rate, max_active_rate)
        # split_annotations_into_clips(['csgo'], min_clip_duration, clip_duration, overlap, min_active_rate, max_active_rate)
        # split_annotations_into_clips(['soccer'], min_clip_duration, clip_duration, overlap, min_active_rate, max_active_rate)
        # split_annotations_into_clips(['black_myth_wukong'], min_clip_duration, clip_duration, overlap, min_active_rate, max_active_rate)
        # split_annotations_into_clips(['cyberpunk'], min_clip_duration, clip_duration, overlap, min_active_rate, max_active_rate)



        split_annotations_into_clips(['livecc_v2'], min_clip_duration, clip_duration, overlap)
        split_annotations_into_clips(['livecc_v3'], min_clip_duration, clip_duration, overlap)
        # split_annotations_into_clips(['minecraft'], min_clip_duration, clip_duration, overlap, min_active_rate, max_active_rate)
