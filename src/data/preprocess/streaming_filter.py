import os
import json

ann_path = '/home/v-weicaiyan/ds/DATA/results/liveccbase_rate_alldata_fullfinetuning_final_30/streaming/liveccbase_rate_alldata_fullfinetuning_final_30_threshold_30_contextlen_16384_standard.jsonl'
ann_path = '/home/v-weicaiyan/ds/DATA/results/streamingvlm/streaming/StreamingVLM_standard.jsonl'
ann_path = '/home/v-weicaiyan/ds/DATA/ann/label/streaming_video_commentary_val_standard_wo_user.jsonl'
max_video_end_list = [600, 1200, 1800, 2400, 3000]

for max_video_end in max_video_end_list:
    output_path = ann_path.replace('.jsonl', f'_maxvideoend_{max_video_end}.jsonl')
    with open(ann_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            if data['end'] <= max_video_end:
                fout.write(json.dumps(data) + '\n')
    print(f'Filtered data saved to {output_path}')

