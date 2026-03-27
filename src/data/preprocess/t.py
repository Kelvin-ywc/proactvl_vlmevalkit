import json
import os
import math
import argparse
import random


def split_text_into_segments(text, n_segments):
    # 将 text 按空格拆分成 n_segments 段，尽量均匀分配单词, 如果segments段数小于n_segments, 则只取实际段数,空格需要保留
    words = text.split()
    # mask = []
    total_words = len(words)
    if total_words == 0:
        return []
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

def sentence_to_seconds(item):
    new_annotations = []
    if item['role'] == 'user' and item['speaker'] == 'user' and 'query' in item:
        new_annotations.append(item)
        return new_annotations
    text = item['text']
    if not text.strip():
        return new_annotations
    duration = item['end'] - item['start']
    text_chunks = split_text_into_segments(text, duration)
    for i, chunk in enumerate(text_chunks):
        start_time = item['start'] + i
        end_time = start_time + 1
        new_item = {
            'speaker': item['speaker'],
            'role': item['role'],
            'start': start_time,
            'end': end_time,
            'text': chunk,
        }
        new_annotations.append(new_item)
    return new_annotations

def split_text_into_seconds(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            new_annotations = []
            for item in ann['annotations']:
                if item['end'] - item['start'] <= 0:
                    continue
                new_annotations.extend(sentence_to_seconds(item))
            ann['annotations'] = new_annotations
            f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')

file_to_save_split_speakers = 'ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/long_video_commentary/long_video_commentary_split_speakers.jsonl'
file_to_save_split_text = file_to_save_split_speakers.replace('split_speakers', 'split_text')
print(f'[Split text] Splitting text into seconds and saving to {file_to_save_split_text}...')
split_text_into_seconds(file_to_save_split_speakers, file_to_save_split_text)