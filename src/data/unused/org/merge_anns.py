'''
This file is used to preprocess custom data for the ProActMLLM project.
There are two types of data preprocessing supported:
1. Streaming commentary data, given the video, the model generates commentary second by second.
2. Instant commentary data, user provides query at the specific time point, the model generates commentary for that query.
The data format is as follows:
{
    "video_path": "",  # Path to the video file
    "video_begin": 0,  # Start time of the video segment
    "video_end": 90,   # End time of the video segment
    "duration": 90,    # Duration of the video segment
    "speakers": ['SPEAKER_00', 'SPEAKER_01'],  # List of speakers in the video
    "history": "(SPEAKER_0): ***.\n(SPEAKER_1): ***.\n(USER): ***.\n(CURRENT_SPEAKER): ***.\n", # optional, context or previous dialogue history
    "annotations": [   # List of annotations for commentary
        {
            "query": "Describe", # optional, when empty, default to recording a commentary segment
            "begin_time": 1,    # Start time for training
            "end_time": 5,      # End time for training
            "commentary": "",   # Commentary content
            "speaker": "SPEAKER_00", # Speaker providing the commentary
            "active": True      # Whether this annotation is active
        },
        {
            "query": "Describe", # optional
            "begin_time": 7,
            "end_time": 10,
            "commentary": "",
            "speaker": "SPEAKER_01",
            "active": False
        }
    ]
},
Then the data will be segmented into smaller clips based on the annotations for training. Each clips will be constrained within VIDEO_DURATION length with VIDEO_OVERLAP between two clips.
Now, we support dataset including:
Game commentary datasets:
- LOL for multiple speakers commentary.
- SoccerNet for single speaker commentary.
- Minecraft for instant commentary.
Custom commentary datasets:
- Livecc for common instant commentary.
'''

DATA_NAME_TO_BASE_ANN = {
    'lol': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/game_commentary/lol/ann/lol_merged.jsonl',
    'csgo': '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_merged.jsonl',
    'cyberpunk': '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_merged.jsonl',
    'black_myth_wukong': '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_merged.jsonl',
    'soccer': '/home/v-weicaiyan/ds/DATA/SoccerNet/commentaries',
    'livecc': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft/videos/videos',
    'livecc_v2': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft_6k_12k/videos/videos',
    'livecc_v3': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft_12k_24k/videos/videos',
    
}
DATA_NAME_TO_VIDEO = {
    'lol': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos',
    'csgo': '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/videos',
    'cyberpunk': '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/videos',
    'black_myth_wukong': '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/videos',
    'soccer': '/home/v-weicaiyan/ds/DATA/SoccerNet/SoccerNet/videos',
    'livecc': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft/videos/videos',
    'livecc_v2': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft_6k_12k/videos/videos',
    'livecc_v3': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft_12k_24k/videos/videos',
}

ANN2SAVE = {
    'livecc': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft/ann',
    'livecc_v2': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft_6k_12k/ann',
    'livecc_v3': '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/live_sft_12k_24k/ann',
}
SAVE_DIR = '/home/v-weicaiyan/ds/DATA/ann'

import json
import os
import ffmpeg
import math

def merge_words_into_seconds(words, video_start, video_end):
    commentary_segments = []
    first_word = words[0]
    first_word_start = math.floor(first_word[0])
    assert first_word_start >= video_start, f'first word start {first_word_start} < video start {video_start}'

    commentary_dict = {
        'begin_time': first_word_start,
        'end_time': first_word_start+1,
        'text': ' ' + first_word[2]
    }
    for word_info in words[1:]:
        word_start = math.floor(word_info[0])
        # 多余的词语直接跳过
        if word_start >= video_end:
            break
        if commentary_dict is None:
            commentary_dict = {
                'begin_time': word_start,
                'end_time': word_start+1,
                'text': ' ' + word_info[2]
            }
        elif word_start == commentary_dict['begin_time']:
            commentary_dict['text'] += ' ' + word_info[2]
        else:
            # 保存上一个commentary_dict
            commentary_segments.append(commentary_dict)
            # 创建新的commentary_dict
            commentary_dict = {
                'begin_time': word_start,
                'end_time': word_start+1,
                'text': ' ' + word_info[2]
            }
    # 保存最后一个commentary_dict
    if commentary_dict is not None:
        commentary_segments.append(commentary_dict)
    return commentary_segments

def preprocess_livecc(ann_base_dir, video_dir, save_f):
    # Implement the preprocessing logic for Livecc dataset
    # 遍历ann_base_dir下的所有json文件
    for root, _, files in os.walk(ann_base_dir):
        print(len(files))
        for ann_file in files:
            # print(ann_file)
            if not ann_file.endswith('.json'):
                continue
            ann_path = os.path.join(ann_base_dir, ann_file)
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            # video_path = ann_data[0]['content'][0]['video']
            video_path = os.path.basename(ann_file).replace('.json', '.mp4')
            video_abs_path = os.path.join(video_dir, video_path)
            try:
                probe = ffmpeg.probe(video_abs_path)
            except Exception as e:
                print(f"Error probing video file {video_abs_path}: {e}")
                continue
            video_start = math.floor(ann_data[0]['content'][0]['video_start'])
            video_end = math.floor(float(probe['format']['duration']) + video_start)
            # video_duration = math.ceil(float(probe['format']['duration'])+video_start)
            # video_end = math.ceil(ann_data[0]['content'][0]['video_end'])
            # video_end = min(video_end, video_duration)
            video_duration = video_end - video_start
            assert int(video_duration) == int(float(probe['format']['duration'])), f"video duration mismatch: {video_duration} vs {probe['format']['duration']}"
            # if video_duration >= 60:
            #     continue  # skip videos longer than 1 minute
            # print(ann_data)
            data_to_save = {
                'video_path': video_path,
                'video_begin': video_start,
                'video_end': video_end,
                'duration': video_duration,
                'speakers': ['SPEAKER_0'],
                'history': ann_data[0]['content'][1]['previous'],
                'annotations': [{
                    'query': ann_data[0]['content'][1]['text'],
                    'begin_time': video_start,
                    'end_time': video_end,
                    'commentary': merge_words_into_seconds(ann_data[1]['content'][0]['text_stream'], video_start, video_end),
                    'speaker': 'SPEAKER_0',
                }],
                'metadata': {
                    'title': ann_data[0]['content'][1]['title'],
                    'category': ann_data[0]['content'][1]['category']
                }
            }
            # print(data_to_save)

            save_f.write(json.dumps(data_to_save) + '\n')

def preprocess_soccer(ann_base_dir, video_dir, save_f):
    for file in os.listdir(ann_base_dir):
        if not file.endswith('.json'):
            continue
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
        video_path = os.path.join(eles[0], eles[1], ann_file_name, f'{video_index}_720p.mkv')
        video_abs_path = os.path.join(video_dir, video_path)
        
        ann_file_path = os.path.join(ann_base_dir, file)
        with open(ann_file_path, 'r') as ann_file:
            ann = json.load(ann_file)
        # print(ann)
        # return
        try:
            probe = ffmpeg.probe(video_abs_path)
            video_duration = math.floor(float(probe["format"]["duration"]))
        except Exception as e:
            print(f"Error probing video file {video_abs_path}: {e}")
            continue
        data_to_save = {
            'video_path': video_path,
            'video_begin': 0,
            'video_end': video_duration,
            'duration': video_duration,
            'speakers': ['SPEAKER_0'],
            'history': '',
            'annotations': [],
        }
        for cur_ann in ann:
            cur_ann_to_save = {
                'query': '',
                # 考虑到可能存在连续样本，这里选择begin_time和end_time都使用round
                'begin_time': round(cur_ann['offset']),
                'end_time': min(round(cur_ann['offset'] + cur_ann['duration']), video_duration),
                'commentary': cur_ann['commentary'],
                'speaker': 'SPEAKER_0',
            }
            data_to_save['annotations'].append(cur_ann_to_save)

        save_f.write(json.dumps(data_to_save) + '\n')

def preprocess_lol(ann_base_file, video_dir, save_f):
    with open(ann_base_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # print(data)
            video_path = data['video_path']
            video_abs_path = os.path.join(video_dir, video_path)
            try:
                probe = ffmpeg.probe(video_abs_path)
                video_duration = math.floor(float(probe["format"]["duration"]))
            except Exception as e:
                print(f"Error probing video file {video_abs_path}: {e}")
                continue
            data_to_save = {
                'video_path': video_path,
                'video_begin': 0,
                'video_end': video_duration,
                'duration': video_duration,
                'speakers': data['speakers'],
                'history': '',
                'annotations': [],
            }
            for cur_ann in data['annotations']:
                cur_ann_to_save = {
                    'query': '',
                    'begin_time': round(cur_ann['start']),
                    'end_time': min(round(cur_ann['end']), video_duration),
                    'commentary': cur_ann['text'],
                    'speaker': cur_ann['speaker'],
                }
                data_to_save['annotations'].append(cur_ann_to_save)
            save_f.write(json.dumps(data_to_save) + '\n')


def preprocess_into_standard_format(data_names):
    for data_name in data_names:
        video_dir = DATA_NAME_TO_VIDEO[data_name]
        ann_base_dir = DATA_NAME_TO_BASE_ANN[data_name]

        if data_name in ['lol', 'csgo', 'cyberpunk', 'black_myth_wukong']:
            save_path = ann_base_dir.replace('_merged', '_standard_format')
        elif data_name == 'soccer':
            # save_path = '/home/v-weicaiyan/ds/DATA/SoccerNet/ann'
            save_dir = '/home/v-weicaiyan/ds/DATA/SoccerNet/ann'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{data_name}_standard_format.jsonl'
        elif 'livecc' in data_name:
            # save_dir = f'{ann_base_dir}/commentaries'
            # save_dir = '/home/v-weicaiyan/ds/DATA/live_sft/ann'
            save_dir = ANN2SAVE[data_name]
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{data_name}_standard_format.jsonl'

        with open(save_path, 'w') as save_f:
            if data_name in ['lol', 'csgo', 'cyberpunk', 'black_myth_wukong']:
                preprocess_lol(ann_base_dir, video_dir, save_f)
            elif data_name == 'soccer':
                preprocess_soccer(ann_base_dir, video_dir, save_f)
            elif 'livecc' in data_name:
                print(f'Processing livecc: {ann_base_dir}, {video_dir}')
                preprocess_livecc(ann_base_dir, video_dir, save_f)
            else:
                print(f'Unknown data name: {data_name}')

def peek_one_sample(save_path, idx=0):
    with open(save_path, 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                data = json.loads(line)
                print(json.dumps(data, indent=4))
                return

if __name__ == '__main__':
    # First, preprocess each type of dataset into the same format.
    # preprocess_into_standard_format(['lol', 'csgo', 'cyberpunk', 'black_myth_wukong'])
    # preprocess_into_standard_format(['lol'])
    # preprocess_into_standard_format(['csgo'])
    # preprocess_into_standard_format(['cyberpunk'])
    # preprocess_into_standard_format(['black_myth_wukong'])


    preprocess_into_standard_format(['livecc_v3'])
    # preprocess_into_standard_format(['soccer'])

    # ???????? preprocess_into_standard_format(['minecraft'])
    # preprocess_into_standard_format(['lol', 'soccer', 'livecc'])
    # preprocess_into_standard_format(['livecc'])
    # preprocess_into_standard_format(['soccer'])
    # preprocess_into_standard_format(['lol'])
    # lol_save_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/lol_standard_format.jsonl'
    # peek_one_sample(lol_save_path, idx=10)
    # preprocess_into_standard_format(['csgo'])
    # csgo_save_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one/csgo_standard_format.jsonl'
    # peek_one_sample(csgo_save_path, idx=10)
    # check files, ensure each duration > 0.

# ['lol', 'csgo', 'cyberpunk', 'black_myth_wukong', 'soccer', 'livecc'] merge -> standard format -> train/val/test split -> split speakers -> split seconds -> split clips
# ['soccer'] -> standard format -> train/val/test -> split speakers -> split seconds -> split clips
# ['minecraft'] -> split speakers -> train/val/test -> split seconds -> split clips
    