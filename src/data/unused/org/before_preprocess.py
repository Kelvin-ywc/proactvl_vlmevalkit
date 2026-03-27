import os
from pathlib import Path
import decord
import math
import json
import ffmpeg
import re


def split_video_train_val_test(ann_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 如果存在二级目录，则尽量确保train/val/test二级目录在同一个subset中
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    total_videos = len(data)
    val_count = int(total_videos * val_ratio)
    test_count = int(total_videos * test_ratio)
    train_count = total_videos - val_count - test_count
    train_set = data[:train_count]
    val_set = data[train_count:train_count + val_count]
    test_set = data[train_count + val_count:]
    # 直接保存,后缀为 .train.jsonl, .val.jsonl, .test.jsonl
    base_path = os.path.splitext(ann_file)[0]
    with open(f"{base_path}_train.jsonl", 'w', encoding='utf-8') as f_train:
        for item in train_set:
            f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(f"{base_path}_val.jsonl", 'w', encoding='utf-8') as f_val:
        for item in val_set:
            f_val.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(f"{base_path}_test.jsonl", 'w', encoding='utf-8') as f_test:
        for item in test_set:
            f_test.write(json.dumps(item, ensure_ascii=False) + '\n')   

def to_mp4_name(s: str) -> str:
    # 把结尾的 ".webm"（可选地前面带一个 ".mp4"）替换成 ".mp4"
    if 'webm' in s:
        return re.sub(r'(?i)(?:\.mp4)?\.webm$', '.mp4', s)
    elif 'mkv' in s:
        return re.sub(r'(?i)(?:\.mp4)?\.mkv$', '.mp4', s)
    return s

# 对于爬取的lol数据，首先清洗一遍

def merge_speakers_and_smooth_timestep_in_one_file(input_dir, video_dir, output_jsonl):
    # 遍历所有的 JSON 文件，合并 speaker 字段，并平滑时间戳
    # 1. 如果存在二级目录，遍历input_dir 下的所有文件夹，json文件存放在每个文件夹下，否则直接遍历 input_dir 下的所有 json 文件
    ann_list = []
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in Path(folder_path).glob('*.json'):
            # 跳过元数据
            if 'role_metadata.json' in str(file):
                continue
            print(f"[info] Processing file: {file}")    
            # 推导视频相对路径与绝对路径
            video_rel_path = os.path.join(folder, file.name).replace('.json', '.mp4')
            cur_video_path = os.path.join(video_dir, video_rel_path)
            print(f"[merge] video: {cur_video_path}")

            # 读取视频元信息
            try:
                probe = ffmpeg.probe(cur_video_path)
                # 某些视频文件格式可能存在问题，导致无法读取正确的时长，这里将所有的视频时长向下取整
                # video_duration = math.floor(float(probe['streams'][0]['duration']))
                video_duration = math.floor(float(probe['format']['duration']))
            except Exception as e:
                print(f"[error] Error reading video file: {cur_video_path}, {e}")
                continue

            # 读取标注并组装
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[error] Bad json: {file} ({e})")
                continue
            # print(data)
            # 合并 speaker 字段, 平滑时间戳, 组装结果
            current_ann = {
                "video_path": video_rel_path,
                "video_duration": video_duration,
                "speakers": [],
                "annotations": []
            }
            for speaker_data in data:
                speaker = speaker_data['Speaker']
                assert speaker not in current_ann['speakers'], f"Duplicate speaker {speaker} in file {file}"
                current_ann['speakers'].append(speaker)

                for conversation in speaker_data['Conversation']:
                    conversation['speaker'] = speaker
                    current_ann['annotations'].append(conversation)
            # current_ann['annotations'] 按照 start 时间排序
            current_ann['annotations'].sort(key=lambda x: x.get('start', 0))
            # 2. 对每个文件夹下的所有json文件，进行处理，合并 speaker 字段，并平滑时间戳
            # 平滑时间戳, 如果下一个标签开始时间小于上一个标签结束时间，则将两个时间调整为两个时间的均值
            print(f"[info] Smoothing timestamps for file: {file}")
            for i in range(1, len(current_ann['annotations'])):
                prev_ann = current_ann['annotations'][i-1]
                curr_ann = current_ann['annotations'][i]
                prev_end = prev_ann.get('end', 0)
                curr_start = curr_ann.get('start', 0)
                if curr_start < prev_end:
                    avg_time = (prev_end + curr_start) / 2
                    prev_ann['end'] = avg_time
                    curr_ann['start'] = avg_time
            ann_list.append(current_ann)


    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    # 3. 将处理后的结果保存到 output_jsonl中
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for ann in ann_list:
            f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')

def time2seconds(time_str):
    """Convert time string 'HH:MM:SS' to total seconds."""
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

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

def split_into_seconds(sub_annotations, shift_second):
    new_sub_annotations = []
    for ann in sub_annotations:

        begin_time = time2seconds(ann['begin_time'])
        end_time = time2seconds(ann['end_time'])
        speaker = ann['speaker']
        commentary = ann['commentary']
        if begin_time >= end_time:
            print(f"[warning] begin_time >= end_time for annotation: {ann}")
            continue
        text_clips = split_text_into_segments(commentary, end_time - begin_time)
        for idx, text_clip in enumerate(text_clips):
            # {"begin_time": 0, "end_time": 1, "text": " Okay,", "speaker": "SPEAKER_0", "role": "assistant"}
            new_sub_annotations.append({
                "begin_time": begin_time - shift_second + idx,
                "end_time": begin_time - shift_second + idx + 1,
                "speaker": speaker,
                "commentary": text_clip,
                'role': 'assistant'
            })
    return new_sub_annotations


def split_minecraft_speakers(input_json, output_jsonl, shift_second):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[error] Bad json: {input_json} ({e})")
            return
        print(len(data))
        for item in data:
            speakers = item['speakers']
            for speaker in speakers:        
                video_path = item['video_path']
                video_begin = math.floor(item['video_begin'])
                video_end = math.floor(item['video_end'])
                duration = video_end - video_begin

                ann = {
                    "video_path": to_mp4_name(video_path),
                    "video_begin": video_begin,
                    "video_end": video_end,
                    "duration": duration,
                    "speakers": speakers,
                    "active_speaker": speaker,
                    "annotations": []
                }
                for conversation in item['annotations']:
                    query = conversation['query']
                    sub_annotations = conversation['sub_annotations']
                    begin_time = time2seconds(sub_annotations[0]['begin_time'])
                    speaker = sub_annotations[0]['speaker']
                    if begin_time < shift_second:
                        print(f"[warning] Skipping annotation starting before shift time: {begin_time} in video {video_path}")
                        continue
                    ann['annotations'].append({
                        "begin_time": begin_time - shift_second,
                        'end_time': begin_time - shift_second + 1,
                        "query": query,
                        'role': 'user',
                        'history': '',
                    })
                    if speaker != ann['active_speaker']:
                        raise ValueError(f"Speaker mismatch: {speaker} != {ann['active_speaker']}")
                    new_anns = split_into_seconds(sub_annotations, shift_second)
                    ann['annotations'].extend(new_anns)
                # 将处理后的结果保存到 output_jsonl中
                f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')


    #     video_rel_path = item['video_path']
    #     video_duration = item['video_duration']
    #     speakers = item['speakers']
    #     annotations = item['annotations']

    #     current_ann = {
    #         "video_path": video_rel_path,
    #         "video_duration": video_duration,
    #         "speakers": [],
    #         "annotations": []
    #     }
    #     for speaker_data in speakers:
    #         speaker = speaker_data['name']
    #         assert speaker not in current_ann['speakers'], f"Duplicate speaker {speaker} in file {input_json}"
    #         current_ann['speakers'].append(speaker)

    #     for conversation in annotations:
    #         conversation['speaker'] = conversation.get('speaker', 'unknown')
    #         current_ann['annotations'].append(conversation)
        
    #     # current_ann['annotations'] 按照 start 时间排序
    #     current_ann['annotations'].sort(key=lambda x: x.get('start', 0))
    #     ann_list.append(current_ann)

    # os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    # # 将处理后的结果保存到 output_jsonl中
    # with open(output_jsonl, 'w', encoding='utf-8') as f_out:
    #     for ann in ann_list:
    #         f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')
if __name__ == "__main__":
    # # for lol
    # ann_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/raw'
    # video_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos'
    # save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_merged.jsonl'
    # merge_speakers_and_smooth_timestep_in_one_file(ann_dir, video_dir, save_jsonl)

    # # for csgo
    # ann_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/raw'
    # video_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/videos'
    # save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_merged.jsonl'
    # merge_speakers_and_smooth_timestep_in_one_file(ann_dir, video_dir, save_jsonl)

    # 赛博朋克
    # ann_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/raw'
    # video_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/videos'
    # save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_merged.jsonl'
    # merge_speakers_and_smooth_timestep_in_one_file(ann_dir, video_dir, save_jsonl)

    # 黑猴
    # ann_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/raw'
    # video_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/videos'
    # save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_merged.jsonl'
    # merge_speakers_and_smooth_timestep_in_one_file(ann_dir, video_dir, save_jsonl)

    # for minecraft
    ann_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/minecraft/raw/convert_minecraft_data.json'
    save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/minecraft/ann/minecraft_split_speaker.jsonl'
    shift_second = 0
    split_minecraft_speakers(ann_dir, save_jsonl, shift_second)

    # livecc
    
    # soccernet