import subprocess
from pathlib import Path

def remux_or_transcode(src: Path, dst: Path) -> str:
    """先尝试无损封装到MP4，失败则转码到H.264 + AAC。返回 'remuxed' 或 'transcoded'。"""
    # 1) 无损封装（快、无画质损失）
    cmd1 = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c", "copy", str(dst)]
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if r1.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return "remuxed"

    # 2) 兼容性转码（较慢、会重新编码）
    cmd2 = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        str(dst)
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode == 0:
        return "transcoded"

    raise RuntimeError(f"ffmpeg failed:\n{r1.stderr}\n{r2.stderr}")

def output_mp4_path(p: Path) -> Path:
    name_lower = p.name.lower()
    if name_lower.endswith(".mp4.mkv"):
        # 去掉末尾 .mkv，变成真正的 xxx.mp4
        return p.with_name(p.name[:-4])
    else:
        # 普通 .mkv -> .mp4
        return p.with_suffix(".mp4")

def batch_convert(root="./"):
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if name_lower.endswith(".mkv") or name_lower.endswith(".mp4.mkv"):
            dst = output_mp4_path(p)
            if dst.exists():
                print(f"Skip (exists): {dst.name}")
                continue
            mode = remux_or_transcode(p, dst)
            print(f"{p.name} -> {dst.name} ({mode})")

if __name__ == "__main__":
    batch_convert("/home/v-weicaiyan/ds/DATA/game_commentary/minecraft/videos")  # 改成你的目录
