# python -m src.data.preprocess.add_video_dir --input_file DATA/ann/livecc_final_train.jsonl --output_file DATA/ann/livecc_final_train_with_videodir.jsonl --video_dir DATA/live_sft/videos

ann_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/livecc_final_train_org.jsonl'
save_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/livecc_final_train.jsonl'


import json
import os

with open(ann_path, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        video_file = os.path.basename(data['video_path'])
        data['video_path'] = os.path.join('live_sft/videos', video_file)
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')