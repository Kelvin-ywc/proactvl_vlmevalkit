import os
from pathlib import Path
import decord
import math
import json
import ffmpeg

'''

[
    {
        "Speaker": "SPEAKER_00",
        "Conversation": [
            {
                "text": "Me?",
                "start": 0.031,
                "end": 0.395
            }
        ]
    },
    {
        "Speaker": "SPEAKER_01",
        "Conversation": [
            {
                "text": "Yes, you.",
                "start": 0.395,
                "end": 1.23
            }
        ]
    }
]
->
[   
    {
        "Speakers": ["SPEAKER_00", "SPEAKER_01"],
        "video_path": "path/to/video.mp4",
        "video_duration": 123,
        "Conversation": [
            {
                "text": "Me?",
                "commentary": "Me?",
                "start": 0.031,
                "end": 0.395,
                "begin_time_int": 0,
                "end_time_int": 0,
                "duration": 0,
                "speaker": "SPEAKER_00"
            },
            {
                "text": "Yes, you.",
                "start": 0.395,
                "end": 1.23,
                "speaker": "SPEAKER_01"
            }
        ]
    },
]

'''
def merge_speakers_and_smooth_timestep_in_one_file(input_dir, video_dir, output_jsonl):
    # 遍历所有的 JSON 文件，合并 speaker 字段，并平滑时间戳
    # 1. 遍历input_dir 下的所有文件夹，json文件存放在每个文件夹下
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
                video_duration = math.floor(float(probe['streams'][0]['duration']))
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
                speaker = speaker_data.get('Speaker', 'UNKNOWN')
                if speaker not in current_ann['speakers']:
                    current_ann['speakers'].append(speaker)
                for conversation in speaker_data.get('Conversation', []):
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
            # 添加begin_time_int, end_time_int,用于训练
            print(f"[info] Adding begin_time_int and end_time_int for file: {file}")
            for ann in current_ann['annotations']:
                begin_time = float(ann.get('start', 0) or 0)
                end_time = float(ann.get('end', 0) or 0)
                ann['begin_time_int'] = round(begin_time)
                ann['end_time_int'] = min(round(end_time), video_duration)
                # ann['duration'] = max(0.0, end_time - begin_time)
                ann['duration'] = max(0.0, ann['end_time_int'] - ann['begin_time_int'])
                if ann['duration'] == 0.0:
                    print(f"[warn] Zero duration annotation in {file}: {ann}")
            ann_list.append(current_ann)
            '''
            二次检查，确保所有duration 均大于0，如果duration等于0，
            首先尝试将end_time_int向后扩展1秒，如果end_time_int达到了结尾或者大约下一个标签的开始时间，则尝试将begin_time_int向前扩展1秒，如果小于0或者小于上一个标签的结束时间，则打印警告并跳过该标签
            ''' 
            print(f"[info] Final check for non-zero durations for file: {file}")
            for idx, ann in enumerate(current_ann['annotations']):
                if ann['duration'] == 0.0:
                    begin_time_int = ann['begin_time_int']
                    end_time_int = ann['end_time_int']
                    # 尝试将end_time_int向后扩展1秒
                    if end_time_int < video_duration:
                        # 检查是否会和下一个标签冲突
                        if idx + 1 < len(current_ann['annotations']):
                            next_ann = current_ann['annotations'][idx + 1]
                            if end_time_int + 1 <= next_ann['begin_time_int']:
                                ann['end_time_int'] += 1
                                ann['duration'] = max(0, ann['end_time_int'] - ann['begin_time_int'])
                                continue
                        else:
                            ann['end_time_int'] += 1
                            ann['duration'] = max(0, ann['end_time_int'] - ann['begin_time_int'])
                            continue
                    # 尝试将begin_time_int向前扩展1秒
                    if begin_time_int > 0:
                        # 检查是否会和上一个标签冲突
                        if idx - 1 >= 0:
                            prev_ann = current_ann['annotations'][idx - 1]
                            if begin_time_int - 1 >= prev_ann['end_time_int']:
                                ann['begin_time_int'] -= 1
                                ann['duration'] = max(0, ann['end_time_int'] - ann['begin_time_int'])
                                continue
                        else:
                            ann['begin_time_int'] -= 1
                            ann['duration'] = max(0, ann['end_time_int'] - ann['begin_time_int'])
                            continue
                    print(f"[warn] Cannot fix zero-length annotation in {file}: {ann}")
                    # 标记为跳过该标签
                    ann['skip'] = True
            # 最后将标记为跳过的标签移除
            current_ann['annotations'] = [ann for ann in current_ann['annotations'] if not ann.get('skip', False)]

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    # 3. 将处理后的结果保存到 output_jsonl中
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for ann in ann_list:
            f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')

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

'''
将commentary 按照每秒进行拆分，首先，对每个annotation按空格拆分成多个词语，
然后根据begin_time_int 和 end_time_int 计算出该annotation 跨越的秒数，
将这些词语均匀分配到这些秒数中，生成新的annotation，
新的annotation 的begin_time_int 和 end_time_int 分别为对应的秒数，
commentary 为分配到该秒数的词语拼接而成,如果单词数小于秒数，则删除多余的秒数
'''
def split_commentary_into_seconds(ann_file, save_file_path):
    with open(ann_file, 'r', encoding='utf-8') as f_in, open(save_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)

            new_annotations = []
            for ann in data['annotations']:
                commentary = ann.get('text', '').strip()
                if commentary == '':
                    continue
                begin_time_int = ann.get('begin_time_int', 0)
                end_time_int = ann.get('end_time_int', 0)
                duration_seconds = end_time_int - begin_time_int
                assert duration_seconds >= 0, f"Invalid duration_seconds: {duration_seconds} for ann: {ann}"
                assert duration_seconds == ann.get('duration', 0), f"Mismatch duration_seconds: {duration_seconds} and ann['duration']: {ann.get('duration', 0)} for ann: {ann}"
                # 将commentary 按照 duration_seconds 拆分成多个部分
                segments = split_text_into_segments(commentary, duration_seconds)
                if len(segments) != duration_seconds:
                    print(f"[warn] Segment count {len(segments)} does not match duration_seconds {duration_seconds} for ann: {ann}")
                for i, segment in enumerate(segments):
                    if i > 0:
                        segment = ' '+segment  # 保留空格
                    new_ann = {
                        "commentary": segment,
                        'begin_time': begin_time_int + i,
                        'end_time': begin_time_int + i + 1,
                        'duration': 1,
                        'speaker': ann.get('speaker', 'UNKNOWN')
                    }
                    # new_ann = ann.copy()
                    # new_ann['commentary'] = segment
                    # new_ann['org_commentary'] = commentary
                    # new_ann['org_begin_time_int'] = begin_time_int
                    # new_ann['org_end_time_int'] = end_time_int
                    # new_ann['org_duration'] = duration_seconds
                    # new_ann['begin_time_int'] = begin_time_int + i
                    # new_ann['end_time_int'] = begin_time_int + i + 1
                    # new_ann['duration'] = 1
                    new_annotations.append(new_ann)
            data['annotations'] = new_annotations
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

def split_commentary_into_speakers(ann_file, save_file_path):
    ann_list = []
    with open(ann_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line)
            speakers = data.get('speakers', [])
            annotations = data.get('annotations', [])
            for speaker in speakers:
                new_row = {
                    "video_path": data.get('video_path', ''),
                    "video_duration": data.get('video_duration', 0),
                    "speakers": speakers,
                    "active_speaker": speaker,
                    "annotations": []
                }
                for ann in annotations:
                    active = ann.get('speaker', 'UNKNOWN') == speaker
                    if active:
                        new_anns = {
                            "begin_time": ann.get('begin_time', 0),
                            "end_time": ann.get('end_time', 0),
                            "duration": ann.get('duration', 0),
                            "commentary": ann.get('commentary', ''),
                            "speaker": ann.get('speaker', 'UNKNOWN'),
                            "active": active
                        }
                    else:
                        new_anns = {
                            "begin_time": ann.get('begin_time', 0)+1,
                            "end_time": ann.get('end_time', 0) + 1,
                            "duration": ann.get('duration', 0),
                            "commentary": ann.get('commentary', ''),
                            "speaker": ann.get('speaker', 'UNKNOWN'),
                            "active": active
                        }
                    new_row['annotations'].append(new_anns)
                ann_list.append(new_row)
    print(f'[info] Total speaker-separated samples generated: {len(ann_list)}')
    print(f"[info] Saving speaker-separated samples to {save_file_path}")
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w', encoding='utf-8') as f_out:
        for ann in ann_list:
            f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')

'''
处理split_commentary_into_speakers的返回结果，根据视频的duration，将commentary 按照clip_duration进行拆分，允许overlap秒的重叠
'''
def split_commentary_into_clips(ann_file, clip_duration, overlap, save_file_path):
    with open(ann_file, 'r', encoding='utf-8') as f_in, open(save_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            video_duration = data.get('video_duration', 0)
            annotations = data.get('annotations', [])
            clip_start = 0
            while clip_start < video_duration:
                clip_end = min(clip_start + clip_duration, video_duration)
                new_row = {
                    "video_path": data.get('video_path', ''),
                    "video_duration": video_duration,
                    "speakers": data.get('speakers', []),
                    "active_speaker": data.get('active_speaker', ''),
                    "video_begin": clip_start,
                    "video_end": clip_end,
                    "annotations": []
                }
                for ann in annotations:
                    ann_begin = ann.get('begin_time', 0)
                    ann_end = ann.get('end_time', 0)
                    # 检查annotation 是否在当前clip范围内
                    if ann_end <= clip_start or ann_begin >= clip_end:
                        continue
                    # 计算新的annotation的begin_time和end_time
                    new_begin = max(ann_begin, clip_start)
                    new_end = min(ann_end, clip_end)
                    new_duration = new_end - new_begin
                    if new_duration <= 0:
                        continue
                    new_ann = {
                        "begin_time": new_begin,
                        "end_time": new_end,
                        "duration": new_duration,
                        "commentary": ann.get('commentary', ''),
                        "speaker": ann.get('speaker', 'UNKNOWN'),
                        "active": ann.get('active', False)
                    }
                    new_row['annotations'].append(new_ann)
                f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')
                clip_start += (clip_duration - overlap)
                
def peek_one_sample(ann_file, idx):
    if idx < 0:
        # 读取最后一行
        with open(ann_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            data = json.loads(lines[-1])
        print(json.dumps(data, indent=4, ensure_ascii=False))
        return
    with open(ann_file, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            print(f"[info] Reading line {i}")
            if i == idx:
                data = json.loads(line)
        print(json.dumps(data, indent=4, ensure_ascii=False))

def check_ann(ann_file):
    with open(ann_file, 'r', encoding='utf-8') as f_in:
        min_duration = float('inf')
        for idx, line in enumerate(f_in):
            data = json.loads(line)
            video_duration = data.get('video_duration', 0)
            annotations = data.get('annotations', [])
            video_begin = data.get('video_begin', 0)
            video_end = data.get('video_end', video_duration)
            # 首先视频时间要大于等于1秒
            if video_begin < 0 or video_end > video_duration or video_begin >= video_end or video_duration < 1:
                print(f"[error] Invalid video clip time range: {video_begin}-{video_end} in video: {data.get('video_path', '')}")
            if video_end-video_begin < min_duration:
                min_duration = video_end-video_begin
            # 检查所有的annotation的时间戳是否合法，begin_time 和 end_time 非负且不超过video_duration，end_time - begin_time == duration
            for ann in annotations:
                # print(ann)
                begin_time = ann.get('begin_time', 0)
                end_time = ann.get('end_time', 0)
                duration = ann.get('duration', 0)
                if begin_time < 0 or end_time < 0 or duration < 0:
                    print(f"[error] Negative time in annotation: {ann} in video: {data.get('video_path', '')}")
                if end_time > video_duration:
                    print(f"[error] end_time {end_time} exceeds video_duration {video_duration} in annotation: {ann} in video: {data.get('video_path', '')}")
                if end_time - begin_time != duration:
                    print(f"[error] Mismatch duration in annotation: {ann} in video: {data.get('video_path', '')}")
    print(f"[info] Minimum video duration in dataset: {min_duration} seconds")
            
def filter_annotations(ann_file, duration_threshold=30):
    # 过滤掉小于30s的标签
    save_file_path = ann_file.replace('.jsonl', '_filtered.jsonl')
    with open(ann_file, 'r', encoding='utf-8') as f_in, open(save_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            video_duration = data.get('video_duration', 0)
            video_begin = data.get('video_begin', 0)
            video_end = data.get('video_end', video_duration)
            if video_end - video_begin < duration_threshold:
                print(f"[info] Skipping short video clip: {video_begin}-{video_end} in video: {data.get('video_path', '')}")
                continue
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

def describe_ann(ann_file):
    # 统计annotation的基本信息, 包括样本数，视频总时长，annotation标注的视频总时长
    total_samples = 0
    total_video_duration = 0
    total_annotation_duration = 0
    with open(ann_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line)
            total_samples += 1
            video_duration = data.get('video_duration', 0)
            video_begin = data.get('video_begin', 0)
            video_end = data.get('video_end', video_duration)
            total_video_duration += (video_end - video_begin)
            annotations = data.get('annotations', [])
            for ann in annotations:
                duration = ann.get('duration', 0)
                total_annotation_duration += duration
    print(f"[info] Total samples: {total_samples}")
    print(f"[info] Total video duration: {total_video_duration} seconds")
    print(f"[info] Total annotation duration: {total_annotation_duration} seconds")
# [info] Total samples: 10964
# [info] Total video duration: 327080 seconds
# [info] Total annotation duration: 300065 seconds
if __name__ == "__main__":
    ann_dir = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/ProactiveBench/dataset/polish/LOL'
    video_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos'
    save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/processed_annotations/lol_merged.jsonl'
    clip_duration = 120
    overlap = 40
    if True:
        if True:
            merge_speakers_and_smooth_timestep_in_one_file(ann_dir, video_dir, save_jsonl)
        if True:
            split_commentary_into_seconds(save_jsonl, save_jsonl.replace('.jsonl', '_split.jsonl'))
        if True:
            # 分离speaker
            split_commentary_into_speakers(save_jsonl.replace('.jsonl', '_split.jsonl'), save_jsonl.replace('.jsonl', '_split_speakers.jsonl'))

        if True:
            # split into clips
            split_commentary_into_clips(save_jsonl.replace('.jsonl', '_split_speakers.jsonl'), clip_duration, overlap, save_jsonl.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'))
        # filter
        if True:
            filter_annotations(save_jsonl.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'), duration_threshold=clip_duration)
    check_ann(save_jsonl.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s_filtered.jsonl'))
    peek_one_sample(save_jsonl.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s_filtered.jsonl'), 1)
    describe_ann(save_jsonl.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'))

# merge_speakers_and_smooth_timestep_in_one_file确保所有解说是连贯的，但对于训练场景来说，需要将其他commentary的解说内容向后移动一秒钟

if __name__ == '__main1__':
    save_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/processed_annotations/lol_merged.jsonl'
    clip_duration = 60
    overlap = 20
    peek_one_sample(save_jsonl.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'), 1)