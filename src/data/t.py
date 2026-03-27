import json

ann_path = '/home/v-weicaiyan/ds/DATA/ann/long_video_commentary_polished_split_text.jsonl'
save_path = '/home/v-weicaiyan/ds/DATA/ann/streaming_video_commentary_val.jsonl'

MAP = {
    ''
}
with open(ann_path, 'r', encoding='utf-8') as f_in, open(save_path, 'w', encoding='utf-8') as f_out:
    for idx, line in enumerate(f_in):
        data = json.loads(line)

        persona = data.get('active_speaker', {}).get('persona', None)
        if isinstance(persona, list) and len(persona) > 0:
            data['active_speaker']['persona'] = persona[0]
            data['dataset_name'] = data['metadata']['dataset_name']
            if data['dataset_name'] == 'black_myth_wukong':
                data['tag'] = 'Wukong'
            else:
                data['tag'] = data['metadata']['tag']
            data['idx'] = idx
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
