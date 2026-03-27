from src.model.modeling_proact import ProAct_OmniModel, ProActConfig
from bertviz import model_view
from src.utils.proact_process import process_interleave_mm_info


def run():
    # load model
    config = ProActConfig(
        model_name_or_path='Qwen/Qwen2.5-Omni-7B',
        enable_audio_output=False,
        state_threshold=0.5
    )
    ckpt_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/AICompanion/ProActMLLM/trainer_output/exp_amlt_20250925-074108/checkpoint-156'
    SYSTEM_INPUT = 'You are a professional sports commentary Please given comment on the given video.'
    model = ProAct_OmniModel.from_pretrained(config, ckpt_path)
    prompt = f'<|im_start|>system\n{SYSTEM_INPUT}<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><chunk_flag>'
    # input
    input_video_path = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/AICompanion/ProActMLLM/DATA/SoccerNet/videos/SoccerNet/spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid/2_224p.mkv'
    duration = 10
    video_info = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": input_video_path,
                    "video_start": 0,
                    "video_end": duration
                }
            ]
        }
    ]
    audios, images, videos = process_interleave_mm_info(video_info, False, return_video_kwargs=False)
    videos = videos[0:1]
    inputs = model.processor(
        text=prompt, 
        audios=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=False
    ).to(model.device)
    outputs = model(**inputs, output_attentions=True)
    attention = outputs['attentions']
    print(f'attention len: {len(attention)}') # attention len: 32
    print(f'attention[0] shape: {attention[0].shape}') # attention
    tokens = model.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # vis
    model_view(attention, tokens)

if __name__ == "__main__":
    if True:
        from src.utils.utils import debug
        debug(9501)
    run()