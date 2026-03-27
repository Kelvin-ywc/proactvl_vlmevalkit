import os
import json
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
from moviepy.video.tools.subtitles import SubtitlesClip

# --- 1. 参数 ---
session_id = "session_20251022-071003"
video_path = f'/home/v-weicaiyan/ds/DATA/game_commentary/lol/videos/2025MSI_T1_vs_GEN/2025MSI_T1_vs_GEN_game5.mp4'
audio_path = f"/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/infer_output/{session_id}/final_commentary.wav"
json_file_path = f"/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/infer_output/{session_id}/commentary_history.json"
output_path = f"/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/infer_output/{session_id}/2025MSI_T1_vs_GEN_game5_commentary.mp4"
srt_path = f"/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/infer_output/{session_id}/generated_subtitles.srt"

start_time = 0   # 裁剪开始时间
end_time = 560   # 裁剪结束时间


# --- 2. 工具函数 ---
def format_time_srt(seconds: float) -> str:
    """将秒转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


# --- 3. 生成 SRT ---
def create_srt_from_json(json_path, srt_output_path, start_offset: float = 0.0):
    """
    从 JSON 创建 SRT 文件（永久保存）
    - 若 assistant_id 为 None 则跳过；
    - start_offset 会把时间左移，使裁剪后从 0s 对齐。
    """
    print(f"Step 1: Generating SRT from {json_path} ...")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing JSON file: {e}")
        return None

    lines = []
    idx = 1

    for seg in data:
        aid = seg.get('assistant_id')
        if aid is None:
            continue

        text_list = seg.get('text', [])
        if not text_list or not isinstance(text_list, list):
            continue

        text = text_list[0]
        b = float(seg.get('begin_second', 0))
        e = float(seg.get('end_second', 0))
        if e <= b:
            continue

        # 时间左移
        b_adj = b - start_offset
        e_adj = e - start_offset
        if e_adj <= 0:
            continue
        if b_adj < 0:
            b_adj = 0.0

        lines.append(str(idx))
        lines.append(f"{format_time_srt(b_adj)} --> {format_time_srt(e_adj)}")
        lines.append(f"[SPEAKER_{aid}] {text}")
        lines.append("")  # 空行
        idx += 1

    with open(srt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"SRT saved to {srt_output_path} ({idx - 1} entries)")
    return srt_output_path


# --- 4. 视频合成处理 ---
def process_video_with_audio_and_subs(v_path, a_path, srt_path, out_path, start_sec, end_sec):
    """
    步骤 2: 剪裁视频 + 替换音频 + 烧录字幕
    - 字幕自动居中底部 + 描边 + 自动换行
    """
    print("Step 2: Loading video and audio files...")

    with VideoFileClip(v_path) as video_clip:
        video_clip = video_clip.subclipped(start_sec, end_sec)
        w, h = video_clip.size
        print(f"Video clipped to {start_sec}s - {end_sec}s, size={w}x{h}")

        # 加载音频并裁剪
        with AudioFileClip(a_path) as audio_clip:
            audio_clip = audio_clip.subclipped(start_sec, end_sec)
            video_clip = video_clip.with_audio(audio_clip)
            print("Audio track replaced.")

            # 字幕样式生成器
            def subtitle_generator(txt):
                return TextClip(
                    text=txt,
                    font=None,               # 默认系统字体
                    font_size=int(h * 0.04), # 字号占视频高度约4%
                    color='white',
                    method='caption',        # 自动换行
                    size=(int(w * 0.9), None),  # 宽度90%
                    stroke_color='black',
                    stroke_width=2
                )

            # 创建字幕对象
            subtitles = SubtitlesClip(srt_path, make_textclip=subtitle_generator)
            # 居中底部 + 安全边距
            subtitles = subtitles.with_position(("center", 0.92), relative=True)

            try:
                subtitles = subtitles.margin(bottom=int(h * 0.04))
            except Exception:
                pass

            # 合成视频 + 字幕
            final_clip = CompositeVideoClip([video_clip, subtitles])

            # 导出最终视频
            print(f"Step 3: Writing final video to {out_path}...")
            final_clip.write_videofile(
                out_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads=4
            )

    print("✅ Video processing complete.")


# --- 5. 主执行 ---
def main():
    if not os.path.exists(video_path):
        print(f"Error: Input video not found at {video_path}")
        return
    if not os.path.exists(audio_path):
        print(f"Error: Input audio not found at {audio_path}")
        return
    if not os.path.exists(json_file_path):
        print(f"Error: Input JSON not found at {json_file_path}")
        return

    # Step 1: 创建 SRT
    srt_file = create_srt_from_json(json_file_path, srt_path, start_offset=start_time)
    if srt_file is None:
        print("Failed to create SRT file. Aborting.")
        return

    # Step 2: 合成视频
    process_video_with_audio_and_subs(
        video_path,
        audio_path,
        srt_file,
        output_path,
        start_time,
        end_time
    )

    print(f"\n🎬 Successfully completed!\nVideo saved to: {output_path}\nSRT saved to: {srt_file}")


if __name__ == "__main__":
    main()
