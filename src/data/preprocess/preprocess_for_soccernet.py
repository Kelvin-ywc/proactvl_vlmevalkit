import os
import json

PERSONA = [
    'Tone: Neutral and adaptive\nVocabulary: Unrestricted\nRhythm & Pacing: Context-driven',
    'The assistant operates as a default persona without stylistic constraints, allowing responses to emerge naturally from the interaction.',
    'Tone: Unspecified\nVocabulary: Open-ended\nRhythm & Pacing: Variable',
    'The assistant generates responses freely based on input and context, without enforcing any predefined style or expression pattern.',
    'Tone: Neutral\nVocabulary: Free-form\nRhythm & Pacing: Adaptive',
    'The assistant maintains no fixed delivery style, adapting content and structure dynamically as the interaction unfolds.',
    'Tone: Flexible\nVocabulary: General-purpose\nRhythm & Pacing: Naturally varying',
    'The assistant functions as an unconstrained baseline persona, prioritizing relevance while leaving expressive choices open.'
]

SOCCER_DIR_PATH = 'soccernet/videos/'
def preprocess_soccernet(ann_path, new_ann_path):
    with open(os.path.expanduser(ann_path), 'r') as f_in, open(os.path.expanduser(new_ann_path), 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            speaker = ann['speakers'][0]
            new_ann = {
                'video_path': os.path.join(SOCCER_DIR_PATH, ann['video_path']),
                'video_duration': ann['duration'],
                'video_begin': 0,
                'video_end': ann['duration'],
                'speakers': {
                    speaker: {
                        'persona': PERSONA
                    }
                },
                'metadata': {
                    'tag': 'soccernet',
                    'dataset_name': 'soccernet',
                },

                'annotations': []
            }
            for item in ann['annotations']:
                begin_time = item['begin_time']
                end_time = item['end_time']
                text = item['commentary']
                new_item = {
                    'text': text,
                    'start': begin_time,
                    'end': end_time,
                    'speaker': speaker
                }
                new_ann['annotations'].append(new_item)
            f_out.write(json.dumps(new_ann, ensure_ascii=False) + '\n')

ann_path = '~/ds/DATA/SoccerNet/ann/soccer_eval_standard_format.jsonl'
new_ann_path = '~/ds/DATA/soccernet/ann/soccernet_standard_format_val.jsonl'
preprocess_soccernet(ann_path, new_ann_path)

ann_path = '~/ds/DATA/SoccerNet/ann/soccer_test_standard_format.jsonl'
new_ann_path = '~/ds/DATA/soccernet/ann/soccernet_standard_format_test.jsonl'
preprocess_soccernet(ann_path, new_ann_path)