import json
import os
import random
import argparse


def random_select_samples(input_files, output_file, select_nums):
    selected_samples = []
    for input_file, select_num in zip(input_files, select_nums):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                    cur_data = json.loads(line)
                    if cur_data['video_begin'] == 0:
                        data.append(cur_data)
            if select_num >= len(data):
                selected_samples.extend(data)
            elif select_num > 0:
                selected_samples.extend(random.sample(data, min(select_num, len(data))))
            elif select_num == 0:
                continue
            else:
                selected_samples.extend(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f'Finish saving {len(selected_samples)} selected samples to {output_file}')

def main():
    parser = argparse.ArgumentParser(description="Randomly select samples from input files")
    parser.add_argument('--input_files', nargs='+', required=True, help='List of input JSON files')
    parser.add_argument('--select_nums', nargs='+', type=int, required=True, help='Number of samples to select from each file')
    parser.add_argument('--output_file', required=True, help='Output file to save selected samples')
    args = parser.parse_args()
    random_select_samples(args.input_files, args.output_file, args.select_nums)

if __name__ == '__main__':
    main()

