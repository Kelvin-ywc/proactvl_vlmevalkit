from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name_or_path: str = "Qwen/Qwen2.5-Omni-7B"
    # ckpt_path: str = "/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/AICompanion/ProActMLLM/trainer_output/exp_amlt_20250925-074108/checkpoint-156"
    ckpt_path: str = '/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/trainer_output/proact_all_fulltuning_base_liveccbase_final/final'
    enable_audio_output: bool = False
    use_audio_in_video: bool = False
    state_threshold: float = 0.5
    accumulated_response: bool = False
    
settings = Settings()