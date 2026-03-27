'''
数据后处理
livecc+minecraft数据占总数据的20%
实时解说数据占80%，其中，15%的数据active rate在0.0-0.3之间，70%的数据active rate在0.3-0.7之间，15%的数据active rate在0.7-1.0之间
'''

import os
import random

ANN_PATH = {
    'lol': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/ann',
    'csgo': '/home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann',
    'black_myth_wukong': '/home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann',
    'cyberpunk': '/home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann',
    'livecc': '/home/v-weicaiyan/ds/DATA/live_sft/ann',
    'livecc_v2': '/home/v-weicaiyan/ds/DATA/live_sft_6k_12k/ann',
    'livecc_v3': '/home/v-weicaiyan/ds/DATA/live_sft_12k_24k/ann',
    'soccer': '/home/v-weicaiyan/ds/DATA/SoccerNet/ann',
    'minecraft': '/home/v-weicaiyan/ds/DATA/game_commentary/minecraft/ann',
}

ann_dir = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/preprocessed/'

def show_distribution(livecc_ann_paths):
    for path in livecc_ann_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
        print(f'clip num in {path}: {len(lines)}')

def combine_livecc(livecc_ann_paths, livecc_cnt_list, clip_length, data_name, data_type):
    livecc_combined = []
    for path in livecc_ann_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            # 随机选取部分
            lines = random.sample(lines, min(len(lines), livecc_cnt_list.pop(0)))
        livecc_combined.extend(lines)
    print(f'livecc combined clip num: {len(livecc_combined)}')
    save_path = os.path.join(ann_dir, f'{data_name}_{data_type}_split_clips_{clip_length}s_combined.jsonl')
    with open(save_path, 'w') as f:
        for line in livecc_combined:
            f.write(line)

def custom_combine(ann_paths, cnt_list, clip_length, overlap, data_name, data_type):
    combined = []
    for path in ann_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            # 随机选取部分
            lines = random.sample(lines, min(len(lines), cnt_list.pop(0)))
        combined.extend(lines)
    print(f'{data_name} combined clip num: {len(combined)}')
    save_path = os.path.join(ann_dir, f'{data_name}_{data_type}_split_clips_{clip_length}s_overlap{overlap}s_combined.jsonl')
    with open(save_path, 'w') as f:
        for line in combined:
            f.write(line)

def combine_minecraft(minecraft_ann_path, minecraft_cnt_list, clip_length, overlap, data_name, data_type):
    # check minecraft
    minecraft_combined = []
    with open(minecraft_ann_path, 'r') as f:
        lines = f.readlines()
        # 随机选取部分
        lines = random.sample(lines, min(len(lines), minecraft_cnt_list.pop(0)))
    minecraft_combined.extend(lines)
    print(f'minecraft combined clip num: {len(minecraft_combined)}')
    save_path = os.path.join(ann_dir, f'{data_name}_{data_type}_split_clips_{clip_length}s_overlap{overlap}s_combined.jsonl')
    with open(save_path, 'w') as f:
        for line in minecraft_combined:
            f.write(line)

def combine_soccer(soccer_ann_paths, soccer_cnt_list, clip_length, overlap):
    soccer_combined = []
    for path in soccer_ann_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            # 随机选取部分
            lines = random.sample(lines, min(len(lines), soccer_cnt_list.pop(0)))
        soccer_combined.extend(lines)
    print(f'soccer combined clip num: {len(soccer_combined)}')
    save_path = os.path.join(ann_dir, f'soccer_split_clips_{clip_length}s_overlap{overlap}s_combined.jsonl')
    with open(save_path, 'w') as f:
        for line in soccer_combined:
            f.write(line)

def combine_lol(lol_ann_paths, lol_cnt_list, clip_length, overlap):
    # check lol
    lol_combined = []
    for path in lol_ann_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            # 随机选取部分
            lines = random.sample(lines, min(len(lines), lol_cnt_list.pop(0)))
        lol_combined.extend(lines)
    print(f'lol combined clip num: {len(lol_combined)}')
    save_path = os.path.join(ann_dir, f'lol_split_clips_{clip_length}s_overlap{overlap}s_combined.jsonl')
    with open(save_path, 'w') as f:
        for line in lol_combined:
            f.write(line)

def combine_csgo(csgo_ann_paths, csgo_cnt_list, clip_length, overlap):
    # check csgo
    csgo_combined = []
    for path in csgo_ann_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            # 随机选取部分
            lines = random.sample(lines, min(len(lines), csgo_cnt_list.pop(0)))
        csgo_combined.extend(lines)
    print(f'csgo combined clip num: {len(csgo_combined)}')
    save_path = os.path.join(ann_dir, f'csgo_split_clips_{clip_length}s_overlap{overlap}s_combined.jsonl')
    with open(save_path, 'w') as f:
        for line in csgo_combined:
            f.write(line)

if __name__ == "__main__":
    # combine_csgo()
    clip_length = 36
    overlap = 18
    dataset_type = 'train'
    livecc_ann_paths = [os.path.join(ANN_PATH['livecc'], f'livecc_{dataset_type}_split_clips_{clip_length}s.jsonl')]
    livecc_v2_ann_paths = [os.path.join(ANN_PATH['livecc_v2'], f'livecc_v2_{dataset_type}_split_clips_{clip_length}s.jsonl')]
    livecc_v3_ann_paths = [os.path.join(ANN_PATH['livecc_v3'], f'livecc_v3_{dataset_type}_split_clips_{clip_length}s.jsonl')]
    soccer_ann_paths = [
        os.path.join(ANN_PATH['soccer'], f'soccer_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active0-30.jsonl'),
        os.path.join(ANN_PATH['soccer'], f'soccer_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active30-70.jsonl'),
        os.path.join(ANN_PATH['soccer'], f'soccer_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active70-100.jsonl')
    ]
    lol_ann_paths = [
        os.path.join(ANN_PATH['lol'], f'lol_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active0-30.jsonl'),
        os.path.join(ANN_PATH['lol'], f'lol_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active30-70.jsonl'),
        os.path.join(ANN_PATH['lol'], f'lol_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active70-100.jsonl')
    ]
    minecraft_ann_paths = [os.path.join(ANN_PATH['minecraft'], f'minecraft_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s.jsonl')] 
    csgo_ann_paths = [
        os.path.join(ANN_PATH['csgo'], f'csgo_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active0-30.jsonl'),
        os.path.join(ANN_PATH['csgo'], f'csgo_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active30-70.jsonl'),
        os.path.join(ANN_PATH['csgo'], f'csgo_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active70-100.jsonl')
    ]   
    black_myth_wukong_ann_paths = [
        os.path.join(ANN_PATH['black_myth_wukong'], f'black_myth_wukong_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active0-30.jsonl'),
        os.path.join(ANN_PATH['black_myth_wukong'], f'black_myth_wukong_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active30-70.jsonl'),
        os.path.join(ANN_PATH['black_myth_wukong'], f'black_myth_wukong_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active70-100.jsonl')
    ]
    cyberpunk_ann_paths = [
        os.path.join(ANN_PATH['cyberpunk'], f'cyberpunk_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active0-30.jsonl'),
        os.path.join(ANN_PATH['cyberpunk'], f'cyberpunk_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active30-70.jsonl'),
        os.path.join(ANN_PATH['cyberpunk'], f'cyberpunk_{dataset_type}_split_clips_{clip_length}s_overlap{overlap}s_active70-100.jsonl')
    ]
    
    all_ann_paths = livecc_ann_paths + minecraft_ann_paths + soccer_ann_paths + lol_ann_paths + csgo_ann_paths + black_myth_wukong_ann_paths + cyberpunk_ann_paths
    all_ann_paths = livecc_ann_paths + livecc_v2_ann_paths + livecc_v3_ann_paths
    show_distribution(all_ann_paths)
    if dataset_type=='train':
        # =======================Train==============================================
        # clip num in /home/v-weicaiyan/ds/DATA/live_sft/ann/livecc_train_split_clips_36s.jsonl: 1600
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/minecraft/ann/minecraft_train_split_clips_36s_overlap18s.jsonl: 2230
        # clip num in /home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_train_split_clips_36s_overlap18s_active0-30.jsonl: 158
        # clip num in /home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_train_split_clips_36s_overlap18s_active30-70.jsonl: 2541
        # clip num in /home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_train_split_clips_36s_overlap18s_active70-100.jsonl: 2135
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_train_split_clips_36s_overlap18s_active0-30.jsonl: 1998
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_train_split_clips_36s_overlap18s_active30-70.jsonl: 3663
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_train_split_clips_36s_overlap18s_active70-100.jsonl: 1285
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_train_split_clips_36s_overlap18s_active0-30.jsonl: 835
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_train_split_clips_36s_overlap18s_active30-70.jsonl: 2626
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_train_split_clips_36s_overlap18s_active70-100.jsonl: 1037
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_train_split_clips_36s_overlap18s_active0-30.jsonl: 660
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_train_split_clips_36s_overlap18s_active30-70.jsonl: 1969
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_train_split_clips_36s_overlap18s_active70-100.jsonl: 424
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_train_split_clips_36s_overlap18s_active0-30.jsonl: 3078
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_train_split_clips_36s_overlap18s_active30-70.jsonl: 3690
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_train_split_clips_36s_overlap18s_active70-100.jsonl: 1422
        livecc_sample_list = [4700] # 1600
        livecc_v2_sample_list = [4800]
        livecc_v3_sample_list = [9500]
        minecraft_sample_list = [2230] # 2230 -> 2230
        soccer_sample_list = [150,2500,1350] # 158+2541+2135=4834 -> 4000
        lol_sample_list = [500,3000,500] # 1998+3663+1285=6946 -> 4000
        csgo_sample_list = [500,2500,1000] # 835+2626+1037=4498 -> 4000
        black_myth_wukong_sample_list = [640,1960,400] # 660+1969+424=3053 -> 3000
        cyberpunk_sample_list = [1000,2500,500] # 3078+3690+1422=8190 ->4000

        # custom_combine(soccer_ann_paths, soccer_sample_list, clip_length, overlap, 'soccer', dataset_type)
        # custom_combine(lol_ann_paths, lol_sample_list, clip_length, overlap, 'lol', dataset_type)
        # custom_combine(csgo_ann_paths, csgo_sample_list, clip_length, overlap, 'csgo', dataset_type)
        # custom_combine(black_myth_wukong_ann_paths, black_myth_wukong_sample_list, clip_length, overlap, 'black_myth_wukong', dataset_type)
        # custom_combine(cyberpunk_ann_paths, cyberpunk_sample_list, clip_length, overlap, 'cyberpunk', dataset_type)


        combine_livecc(livecc_ann_paths, livecc_sample_list, clip_length, 'livecc', dataset_type)
        combine_livecc(livecc_v2_ann_paths, livecc_v2_sample_list, clip_length, 'livecc_v2', dataset_type)
        combine_livecc(livecc_v3_ann_paths, livecc_v3_sample_list, clip_length, 'livecc_v3', dataset_type)
        # combine_minecraft(minecraft_ann_paths[0], minecraft_sample_list, clip_length, overlap, 'minecraft', dataset_type)
    elif dataset_type=='eval':
        # =======================Eval==============================================
        # clip num in /home/v-weicaiyan/ds/DATA/live_sft/ann/livecc_eval_split_clips_36s.jsonl: 199
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/minecraft/ann/minecraft_eval_split_clips_36s_overlap18s.jsonl: 200
        # clip num in /home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_eval_split_clips_36s_overlap18s_active0-30.jsonl: 30
        # clip num in /home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_eval_split_clips_36s_overlap18s_active30-70.jsonl: 368
        # clip num in /home/v-weicaiyan/ds/DATA/SoccerNet/ann/soccer_eval_split_clips_36s_overlap18s_active70-100.jsonl: 203
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_eval_split_clips_36s_overlap18s_active0-30.jsonl: 143
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_eval_split_clips_36s_overlap18s_active30-70.jsonl: 234
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/lol/ann/lol_eval_split_clips_36s_overlap18s_active70-100.jsonl: 197
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_eval_split_clips_36s_overlap18s_active0-30.jsonl: 29
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_eval_split_clips_36s_overlap18s_active30-70.jsonl: 226
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/csgo/ann/csgo_eval_split_clips_36s_overlap18s_active70-100.jsonl: 108
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_eval_split_clips_36s_overlap18s_active0-30.jsonl: 35
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_eval_split_clips_36s_overlap18s_active30-70.jsonl: 128
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Black_Myth_Wukong/ann/black_myth_wukong_eval_split_clips_36s_overlap18s_active70-100.jsonl: 21
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_eval_split_clips_36s_overlap18s_active0-30.jsonl: 287
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_eval_split_clips_36s_overlap18s_active30-70.jsonl: 445
        # clip num in /home/v-weicaiyan/ds/DATA/game_commentary/Cyberpunk_2077/ann/cyberpunk_eval_split_clips_36s_overlap18s_active70-100.jsonl: 158
        livecc_sample_list = [200] # 199
        livecc_v2_sample_list = [200]
        livecc_v3_sample_list = [400]
        minecraft_sample_list = [200] # 200
        soccer_sample_list = [20, 200, 20] # 30,368,203 -> 20, 200, 20
        lol_sample_list = [20,200,20] # 143,234,197 -> 20, 200, 20
        csgo_sample_list = [20,200,20] # 29，226，108 -> 20, 200, 20
        black_myth_wukong_sample_list = [10, 100, 10] # 35，128，21 -> 10, 100, 10
        cyberpunk_sample_list = [20, 200, 20] # 287，445，158 -> 20, 200, 20

        # custom_combine(soccer_ann_paths, soccer_sample_list, clip_length, overlap, 'soccer', dataset_type)
        # custom_combine(lol_ann_paths, lol_sample_list, clip_length, overlap, 'lol', dataset_type)
        # custom_combine(csgo_ann_paths, csgo_sample_list, clip_length, overlap, 'csgo', dataset_type)
        # custom_combine(black_myth_wukong_ann_paths, black_myth_wukong_sample_list, clip_length, overlap, 'black_myth_wukong', dataset_type)
        # custom_combine(cyberpunk_ann_paths, cyberpunk_sample_list, clip_length, overlap, 'cyberpunk', dataset_type)


        # livecc_sample_list = [900]
        combine_livecc(livecc_ann_paths, livecc_sample_list, clip_length, 'livecc', dataset_type)
        combine_livecc(livecc_v2_ann_paths, livecc_v2_sample_list, clip_length, 'livecc_v2', dataset_type)
        combine_livecc(livecc_v3_ann_paths, livecc_v3_sample_list, clip_length, 'livecc_v3', dataset_type)
        # combine_minecraft(minecraft_ann_paths[0], minecraft_sample_list, clip_length, overlap, 'minecraft', dataset_type)