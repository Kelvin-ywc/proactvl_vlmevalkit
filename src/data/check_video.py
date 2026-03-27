import decord
import math
import ffmpeg
from qwen_omni_utils.v2_5.vision_process import extract_vision_info, fetch_image, fetch_video, process_vision_info
from torchcodec.decoders import VideoDecoder

import debugpy
if False:
    try:
        debugpy.listen(('localhost',9501))
        print(f'debug listen on port 9501')
        debugpy.wait_for_client()
    except Exception as e:
        raise RuntimeError(f"Failed to start debugpy: {e}")
    
video_path = '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4'
# vr = decord.VideoReader(video_path)
# total_frames, video_fps = len(vr), vr.get_avg_fps()
# video_duration = math.floor(total_frames / video_fps)
# print(total_frames, video_fps, video_duration)

probe = ffmpeg.probe(video_path)
print(probe)

# [{'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2130, 'chunk_end': 2131}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2131, 'chunk_end': 2132}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2132, 'chunk_end': 2133}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2133, 'chunk_end': 2134}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2134, 'chunk_end': 2135}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2135, 'chunk_end': 2136}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2136, 'chunk_end': 2137}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2137, 'chunk_end': 2138}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2138, 'chunk_end': 2139}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2139, 'chunk_end': 2140}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2140, 'chunk_end': 2141}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2141, 'chunk_end': 2142}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2142, 'chunk_end': 2143}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2143, 'chunk_end': 2144}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2144, 'chunk_end': 2145}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2145, 'chunk_end': 2146}, {'type': 'video', 'video': '/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_AL_vs_FLY/2025MSI_AL_vs_FLY_game2.mp4', 'video_start': 2130, 'video_end': 2147, 'chunk_start': 2146, 'chunk_end': 2147}]

vision_infos = [{"video": video_path, "video_start": 2130.0, "video_end": 2147.0, 'nframes': 2*int(2147-2130)}]
video_input, video_sample_fps = fetch_video(vision_infos[0], return_video_sample_fps=True)
print(video_input.shape, video_sample_fps)