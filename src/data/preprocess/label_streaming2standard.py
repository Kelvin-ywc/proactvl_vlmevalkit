import os
import json

# 将pred 每30s切片

def stream2standard(ann, save_path):
    total_idx = 0
    new_anns = []
    with open(ann_path, 'r') as fin, open(save_path, 'w') as fout:
        for line in fin:
            ann = json.loads(line)
            sorted_seconds = sorted(ann['pred'].keys())
            sub_pred_list = []
            begin_times = range(0, ann['end'], 30)
            for i in range(len(begin_times)):
                sub_pred_list.append({})
            for sec in sorted_seconds:
                idx_in_list = int(sec) // 30
                sub_pred_list[idx_in_list][sec] = ann['pred'][sec]
            for idx, item in enumerate(sub_pred_list):
                tmp_ann_sub = ann.copy()
                tmp_ann_sub['begin'] = begin_times[idx]
                tmp_ann_sub['end'] = min(tmp_ann_sub['begin'] + 30, ann['end'])
                tmp_ann_sub['pred'] = item
                tmp_ann_sub['idx'] = total_idx
                total_idx += 1
                new_anns.append(tmp_ann_sub)
        for new_ann in new_anns:
            fout.write(json.dumps(new_ann) + '\n')
    print(f'Saved to {save_path}')

def stream2standard4label(ann, save_path):
    total_idx = 0
    new_anns = []
    with open(ann_path, 'r') as fin, open(save_path, 'w') as fout:
        for line in fin:
            ann = json.loads(line)

            sub_pred_list = []
            begin_times = range(0, ann['video_end'], 30)
            for i in range(len(begin_times)):
                sub_pred_list.append([])
            for sub_ann in ann['annotations']:
                idx_in_list = sub_ann['start'] // 30
                sub_pred_list[idx_in_list].append(sub_ann)
            for idx, item in enumerate(sub_pred_list):
                tmp_ann_sub = ann.copy()
                tmp_ann_sub['video_begin'] = begin_times[idx]
                tmp_ann_sub['video_end'] = min(tmp_ann_sub['video_begin'] + 30, ann['video_end'])
                tmp_ann_sub['duration'] = tmp_ann_sub['video_end'] - tmp_ann_sub['video_begin']
                tmp_ann_sub['annotations'] = item
                tmp_ann_sub['idx'] = total_idx
                total_idx += 1
                new_anns.append(tmp_ann_sub)
        for new_ann in new_anns:
            fout.write(json.dumps(new_ann) + '\n')
    print(f'Saved to {save_path}')

ann_path = 'DATA/results/streamingvlm/streaming/StreamingVLM.jsonl'
save_path = 'DATA/results/streamingvlm/streaming/StreamingVLM_standard.jsonl'
stream2standard(ann_path, save_path)

ann_path = 'DATA/results/main/proacta_vl/streaming/liveccbase_rate_alldata_fullfinetuning_final_threshold_50_contextlen_16384.jsonl'
save_path = 'DATA/results/main/proacta_vl/streaming/liveccbase_rate_alldata_fullfinetuning_final_threshold_50_contextlen_16384_standard.jsonl'
stream2standard(ann_path, save_path)
ann_path = '/home/v-weicaiyan/ds/DATA/results/liveccbase_rate_alldata_fullfinetuning_final_30/streaming/liveccbase_rate_alldata_fullfinetuning_final_30_threshold_30_contextlen_16384.jsonl'
save_path = '/home/v-weicaiyan/ds/DATA/results/liveccbase_rate_alldata_fullfinetuning_final_30/streaming/liveccbase_rate_alldata_fullfinetuning_final_30_threshold_30_contextlen_16384_standard.jsonl'
stream2standard(ann_path, save_path)

ann_path = '/home/v-weicaiyan/ds/DATA/ann/label/streaming_video_commentary_val.jsonl'
save_path = '/home/v-weicaiyan/ds/DATA/ann/label/streaming_video_commentary_val_standard.jsonl'
stream2standard4label(ann_path, save_path)