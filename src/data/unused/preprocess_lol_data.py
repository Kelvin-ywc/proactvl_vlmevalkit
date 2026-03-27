# This file is used to preprocess the raw data into a format suitable for training the model.
from pathlib import Path
import decord
import math
import os
import json
import ffmpeg


def merge_json_files_to_jsonl(input_dir, video_dir, output_jsonl):
    """
    将 input_dir 下每个子文件夹中的 *.json 合并为 JSONL：
    每个视频一行，结构：
    {
      "video_path": "...",
      "video_duration": int(秒),
      "annotations": [{offset, duration, begin_time, end_time, commentary}, ...]
    }
    """
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    count_written = 0
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for folder in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            print(f"[merge] Processing folder: {folder}")
            for file in Path(folder_path).glob('*.json'):
                # 跳过元数据
                if 'role_metadata.json' in str(file):
                    continue

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
                for idx, one_data in enumerate(data):
                    if data[idx]['Conversation'] is None or len(data[idx]['Conversation']) == 0:
                        print(f"[warn] No annotations in {file}, idx={idx}")
                        continue
                    row_data = {
                        "video_path": video_rel_path,   # 相对 video_dir 的路径
                        "video_duration": video_duration,
                        "anns": []
                    }
                    last_end_time_int = 0
                    # 注意：原始代码 data[0]['Conversation']，保持一致
                    try:
                        for ann in data[idx]['Conversation']:
                            commentary = (ann.get('text') or '').strip()
                            begin_time = float(ann.get('start', 0) or 0)
                            end_time   = float(ann.get('end', 0) or 0)
                            duration   = max(0.0, end_time - begin_time)
                            offset     = begin_time
                            begin_time_int = round(begin_time)
                            end_time_int = min(round(end_time), video_duration)
                            # check commentary 在 [begin_time_int, end_time_int),共end_time_int - begin_time_int 秒，即end_time_int - begin_time_int个chunk
                            # 其中需要用到的标签，commentary，duration，begin_time_int, end_time_int
                            # 如果当前标签开始时间小于上次结束时间，则将开始时间调整为上次结束时间
                            if begin_time_int < last_end_time_int:
                                print(f"[warn] begin_time < last_end_time in {file}: {ann}")
                                begin_time_int = last_end_time_int
                            # 如果调整后开始时间大于等于结束时间，则跳过该标签
                            if begin_time_int > end_time_int:
                                print(f"[warn] begin_time > end_time in {file}: {ann}")
                                continue
                            # 如果开始时间等于结束时间，说明当前commentary在1秒以内，则尝试扩展
                            if begin_time_int == end_time_int:
                                # 尝试begin_time_int向前扩一秒或者end_time_int向后扩一秒
                                if end_time_int < video_duration:
                                    end_time_int += 1
                                elif begin_time_int > 0:
                                    begin_time_int -= 1
                                else:
                                    print(f"[warn] Cannot fix zero-length annotation in {file}: {ann}")
                                    continue
                            # 清除标签中的空样本
                            if commentary == '':
                                print(f"[warn] Skip empty commentary annotation in {file}: {ann}")
                                continue
                            last_end_time_int = end_time_int
                            row_data['anns'].append({
                                "offset": offset,
                                "duration": end_time_int - begin_time_int, # 这里不能用duration，需要重新计算
                                "begin_time": begin_time,
                                "end_time": end_time,
                                "commentary": commentary,
                                'begin_time_int': begin_time_int,
                                'end_time_int': end_time_int,
                                'speaker': data[idx]['Speaker']
                            })
                    except Exception as e:
                        print(f"[error] Parse annotations failed in {file}: {e}")
                        continue

                    # 写入 JSONL（一行一个样本）
                    fout.write(json.dumps(row_data, ensure_ascii=False) + "\n")
                    count_written += 1

    print(f"[done] Wrote {count_written} lines to {output_jsonl}")

def merge_speaker_and_smooth_timestep(input_dir, save_dir):
    """
    合并 input_dir 下每个子文件夹中的 *.json 的 speaker 字段，并平滑时间戳
    """
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue




    print(f"[done] Speaker merge and time smoothing done.")
def preprocess_proactive_benchmark():
    ann_dir = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/ProactiveBench/dataset/polish/LOL'
    video_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos'
    out_jsonl = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/all_annotations.jsonl'
    save_dir = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/processed_annotations'
    merge_speaker_and_smooth_timestep(ann_dir, save_dir)
    exit()
    merge_json_files_to_jsonl(ann_dir, video_dir, out_jsonl)

if __name__ == "__main__":
    preprocess_proactive_benchmark()
