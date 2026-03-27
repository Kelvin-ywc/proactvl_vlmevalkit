import os
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
import math
import decord
import ffmpeg
from src.data.preprocess_new_data import check_ann, split_commentary_into_seconds, filter_annotations, split_commentary_into_speakers, split_commentary_into_clips, peek_one_sample, describe_ann

def smooth_and_dump_annotations(ann_list, output_jsonl):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for current_ann in ann_list:
            anns = current_ann.get('annotations', [])
            video_duration = current_ann.get('video_duration', 0)
            video_path = current_ann.get('video_path', '')

            # 1) 排序
            anns.sort(key=lambda x: x.get('start', 0))

            # 2) 平滑重叠: 若下一个 start < 上一个 end，取均值并分割
            print(f"[info] Smoothing timestamps for video: {video_path}")
            for i in range(1, len(anns)):
                prev_ann = anns[i - 1]
                curr_ann = anns[i]
                prev_end = float(prev_ann.get('end', 0) or 0)
                curr_start = float(curr_ann.get('start', 0) or 0)
                if curr_start < prev_end:
                    avg_time = (prev_end + curr_start) / 2.0
                    prev_ann['end'] = avg_time
                    curr_ann['start'] = avg_time

            # 3) 生成整秒区间与时长（裁剪到 [0, video_duration]）
            print(f"[info] Adding begin_time_int and end_time_int for video: {video_path}")
            for ann in anns:
                begin_time = float(ann.get('start', 0) or 0)
                end_time = float(ann.get('end', 0) or 0)

                ann['begin_time_int'] = max(0, round(begin_time))
                ann['end_time_int'] = min(video_duration, round(end_time))
                ann['duration'] = max(0.0, ann['end_time_int'] - ann['begin_time_int'])

                if ann['duration'] == 0.0:
                    print(f"[warn] Zero duration annotation in {video_path}: {ann}")

            # 4) 二次检查：尝试修复零时长（先扩 end，再扩 begin），避免与邻居冲突
            print(f"[info] Final check for non-zero durations for video: {video_path}")
            for idx, ann in enumerate(anns):
                if ann['duration'] == 0.0:
                    begin_time_int = ann['begin_time_int']
                    end_time_int = ann['end_time_int']

                    # 尝试向后扩展 end_time_int +1
                    if end_time_int < video_duration:
                        no_conflict = True
                        if idx + 1 < len(anns):
                            next_ann = anns[idx + 1]
                            if end_time_int + 1 > int(next_ann.get('begin_time_int', round(next_ann.get('start', 0) or 0))):
                                no_conflict = False
                        if no_conflict:
                            ann['end_time_int'] += 1
                            ann['duration'] = max(0, ann['end_time_int'] - ann['begin_time_int'])
                            if ann['duration'] > 0:
                                continue

                    # 尝试向前扩展 begin_time_int -1
                    if begin_time_int > 0:
                        no_conflict = True
                        if idx - 1 >= 0:
                            prev_ann = anns[idx - 1]
                            if begin_time_int - 1 < int(prev_ann.get('end_time_int', round(prev_ann.get('end', 0) or 0))):
                                no_conflict = False
                        if no_conflict:
                            ann['begin_time_int'] -= 1
                            ann['duration'] = max(0, ann['end_time_int'] - ann['begin_time_int'])
                            if ann['duration'] > 0:
                                continue

                    print(f"[warn] Cannot fix zero-length annotation in {video_path}: {ann}")
                    ann['skip'] = True

            # 5) 过滤掉 skip，并回写再输出一行 JSON
            current_ann['annotations'] = [a for a in anns if not a.get('skip', False)]
            f_out.write(json.dumps(current_ann, ensure_ascii=False) + '\n')


def agg_ann_in_one_file(video_data_dir_path, ann_dir_path, save_path):
    all_anns = []
    for root, dirs, files in os.walk(ann_dir_path):
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
                _cur_video_path = os.path.join(eles[0], eles[1], ann_file_name, f'{video_index}_224p.mkv')
                cur_video_path = os.path.join(video_data_dir_path, _cur_video_path)
                
                ann_file_path = os.path.join(ann_dir_path, file)
                with open(ann_file_path, 'r') as ann_file:
                    ann = json.load(ann_file)
                    # 读取视频元信息
                    try:
                        probe = ffmpeg.probe(cur_video_path)
                        # 某些视频文件格式可能存在问题，导致无法读取正确的时长，这里将所有的视频时长向下取整
                        video_duration = math.floor(float(probe["format"]["duration"]))
                    except Exception as e:
                        print(f"[error] Error reading video file: {cur_video_path}, {e}")
                        continue
                    anns = []
                    for cur_ann in ann:
                        begin_time = cur_ann['offset']
                        end_time = cur_ann['offset'] + cur_ann['duration']
                        begin_time_int = round(begin_time)
                        # make sure end_time_int does not exceed video duration
                        end_time_int = min(round(end_time), video_duration)
                        # filter out those annotations with zero duration
                        if begin_time_int != end_time_int:
                            anns.append(
                                {
                                    'duration': end_time_int - begin_time_int,
                                    'start': begin_time,
                                    'end': end_time,
                                    'speaker': 'SPEAKER_0',
                                    'text': cur_ann['commentary'],
                                    'begin_time_int': begin_time_int,
                                    'end_time_int': end_time_int
                                }
                            )
                    all_anns.append(
                        {
                            'video_path': _cur_video_path,
                            'video_duration': video_duration,
                            'speakers': ['SPEAKER_0'],
                            'annotations': anns
                        }
                    )
    with open(save_path, 'w') as save_file:
        for ann in all_anns:
            save_file.write(json.dumps(ann) + '\n')

# 整合所有annotation文件到一个jsonl文件中
if __name__ == '__main__':
    video_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet'
    ann_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/commentaries'
    save_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/processed_annotations/annotations_all.jsonl'
    agg_ann_in_one_file(video_dir_path, ann_dir_path, save_path)
    split_commentary_into_seconds(save_path, save_path.replace('.jsonl', '_split.jsonl'))
    split_commentary_into_speakers(save_path.replace('.jsonl', '_split.jsonl'), save_path.replace('.jsonl', '_split_speakers.jsonl'))
    clip_duration = 60
    overlap = 20
    split_commentary_into_clips(save_path.replace('.jsonl', '_split.jsonl'), clip_duration, overlap, save_path.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'))
    filter_annotations(save_path.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'))
    check_ann(save_path.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s_filtered.jsonl'))
    peek_one_sample(save_path.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'), -1)
    describe_ann(save_path.replace('.jsonl', f'_split_{clip_duration}s_overlap{overlap}s.jsonl'))

# [info] Total samples: 5431
# [info] Total video duration: 162638 seconds
# [info] Total annotation duration: 107004 seconds