import os
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
import math
import decord
import ffmpeg

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
                    video_duration = 0
                    # with VideoFileClip(cur_video_path) as clip:
                    #     video_duration = math.floor(clip.duration)
                    probe = ffmpeg.probe(video_path)
                    video_duration = math.floor(probe['streams'][0]['duration'])
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
                                    'offset': cur_ann['offset'],
                                    'duration': cur_ann['duration'],
                                    'begin_time': begin_time,
                                    'end_time': end_time,
                                    'commentary': cur_ann['commentary'],
                                    'entity': cur_ann['entity'],
                                    'begin_time_int': round(begin_time),
                                    'end_time_int': min(round(end_time), video_duration)
                                }
                            )
                    all_anns.append(
                        {
                            'video_path': _cur_video_path,
                            'video_duration': video_duration,
                            'anns': anns
                        }
                    )
    with open(save_path, 'w') as save_file:
        for ann in all_anns:
            save_file.write(json.dumps(ann) + '\n')

def split_one_video_anns_to_clips(anns, clip_duration=60):
    clips = []
    video_path = anns['video_path']
    video_duration = anns['video_duration']
    anns = anns['anns']
    begin_time = 0
    end_time = begin_time + clip_duration
    cur_clip_anns = {
        'video_path': video_path,
        'clip_begin_time': begin_time,
        'clip_end_time': end_time,
        'clip_duration': clip_duration,
        'anns': []
    }
    cur_anns = []
    for ann in anns:
        ann_begin_time = ann['begin_time_int']
        ann_end_time = ann['end_time_int']
        if ann_begin_time >= begin_time and ann_end_time <= end_time:
            cur_anns.append(ann)
        elif ann_end_time > end_time:
            # save current clip
            cur_clip_anns['anns'] = cur_anns
            if len(cur_anns) >= 0:
                clips.append(cur_clip_anns)

            # start a new clip
            begin_time = min(end_time, ann_begin_time)
            end_time = min(begin_time + clip_duration, video_duration)
            assert begin_time < end_time
            cur_clip_anns = {
                'video_path': video_path,
                'clip_begin_time': begin_time,
                'clip_end_time': end_time,
                'clip_duration': end_time - begin_time,
                'anns': []
            }
            cur_anns = []
            if ann_begin_time >= begin_time and ann_end_time <= end_time:
                cur_anns.append(ann)
    # save the last clip
    if len(cur_anns) > 0:
        cur_clip_anns['anns'] = cur_anns
        clips.append(cur_clip_anns)
    return clips
            

# 将每个视频分成90s的clip，要求commentary不能跨clip
def split_videos_anns_to_clips(ann_path, save_path, clip_duration=60):
    all_clips = []

    with open(ann_path, 'r') as ann_file:
        for line in ann_file:
            anns = json.loads(line.strip())
            print(f"Processing video: {anns['video_path']}, duration: {anns['video_duration']}, num_anns: {len(anns['anns'])}")
            one_video_clips = split_one_video_anns_to_clips(anns, clip_duration)
            print(f"Split into {len(one_video_clips)} clips")
            all_clips.extend(one_video_clips)

    with open(save_path, 'w') as save_file:
        for clip in all_clips:
            save_file.write(json.dumps(clip) + '\n')

# 整合所有annotation文件到一个jsonl文件中
if __name__ == '__main1__':
    video_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet'
    ann_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/commentaries'
    save_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/annotations_all.jsonl'
    agg_ann_in_one_file(video_dir_path, ann_dir_path, save_path)

# 将annotions中的完整视频分成多个clip
if __name__ == '__main1__':
    clip_durations = [10, 15, 30, 45, 60, 90] # 90s
    # soccernet
    for clip_duration in clip_durations:
        ann_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/annotations_all.jsonl'
        save_path = f'/home/v-weicaiyan/ds/DATA/SoccerNet/annotations_clips_duration_{clip_duration}.jsonl'
        # lol
        ann_path = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/all_annotations.jsonl'
        save_path = f'/home/v-weicaiyan/ds/DATA/game_commentary/lol/annotations_clips_duration_{clip_duration}.jsonl'

        split_videos_anns_to_clips(ann_path, save_path, clip_duration)

# 检查所有的视频片段都是可读的
# if __name__ == '__main__':
#     ann_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/annotations_clips_duration_30.jsonl'
#     video_data_dir_path='/home/v-weicaiyan/ds/DATA/SoccerNet/videos/SoccerNet'
#     with open(ann_path, 'r') as ann_file:
#         for line in ann_file:
#             clip_anns = json.loads(line.strip())
#             video_path = os.path.join(video_data_dir_path, clip_anns['video_path'])
#             clip_begin_time = clip_anns['clip_begin_time']
#             clip_end_time = clip_anns['clip_end_time']
#             try:
#                 with VideoFileClip(video_path) as clip:
#                     subclip = clip.subclip(clip_begin_time, clip_end_time)
#                     print(f"Video {video_path} subclip {clip_begin_time}-{clip_end_time} is readable, duration: {subclip.duration}")
#             except Exception as e:
#                 print(f"Error reading video {video_path} subclip {clip_begin_time}-{clip_end_time}: {e}")

# save pure ann file, filter unnecessary fields
'''
{
    "video_path": "",
    "video_begin": 0,
    "video_end": 90,
    "duration": 90,
    "annotations": [
        {
            "begin_time": 1,
            "end_time": 12,
            "commentary": "",
        },
        {
            "begin_time": 15,
            "end_time": 18,
            "commentary": "",
        },
    ]
}
'''
if __name__ == '__main__':
    # soccernet
    duration_list = [10, 15, 30, 45, 60, 90]
    for duration in duration_list:
        ann_path = f'/home/v-weicaiyan/ds/DATA/SoccerNet/annotations_clips_duration_{duration}.jsonl'
        save_path = f'/home/v-weicaiyan/ds/DATA/SoccerNet/annotations_clips_duration_{duration}_pure.jsonl'
        # lol
        ann_path = f'/home/v-weicaiyan/ds/DATA/game_commentary/lol/annotations_clips_duration_{duration}.jsonl'
        save_path = f'/home/v-weicaiyan/ds/DATA/game_commentary/lol/annotations_clips_duration_{duration}_pure.jsonl'

        with open(ann_path, 'r') as ann_file, open(save_path, 'w') as save_file:
            for line in ann_file:
                clip_anns = json.loads(line.strip())
                pure_anns = {
                    'video_path': clip_anns['video_path'],
                    'video_begin': clip_anns['clip_begin_time'],
                    'video_end': clip_anns['clip_end_time'],
                    'duration': clip_anns['clip_duration'],
                    'annotations': []
                }
                for ann in clip_anns['anns']:
                    pure_anns['annotations'].append(
                        {
                            'begin_time': ann['begin_time_int'],
                            'end_time': ann['end_time_int'],
                            'commentary': ann['commentary']
                        }
                    )
                save_file.write(json.dumps(pure_anns) + '\n')

