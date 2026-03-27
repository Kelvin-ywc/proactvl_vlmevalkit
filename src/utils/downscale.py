import os
import cv2
import argparse


def open_writer(out_path, fps, w, h, preferred=("mp4v", "avc1", "H264")):
    """
    尝试几种常见编码器 fourcc，尽量保证在不同环境下都能写出 mp4。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for code in preferred:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, code
        writer.release()

    raise RuntimeError(
        "无法创建 VideoWriter。你可以尝试：\n"
        "1) 换输出后缀为 .avi 并用 'XVID'\n"
        "2) 或安装带 H264 的 OpenCV / 系统编码器\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Downscale a video and export a new one.")
    parser.add_argument("--in", dest="in_path", required=True, help="input video path")
    parser.add_argument("--out", dest="out_path", required=True, help="output video path, e.g. out.mp4")
    parser.add_argument("--width", type=int, default=0, help="target width (0 means auto)")
    parser.add_argument("--height", type=int, default=0, help="target height (0 means auto)")
    parser.add_argument("--scale", type=float, default=0.0, help="scale factor, e.g. 0.5 (ignored if width/height set)")
    parser.add_argument("--keep_aspect", action="store_true", help="keep aspect ratio when width/height set")
    parser.add_argument("--interp", default="area", choices=["area", "linear", "cubic"], help="resize interpolation")

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.in_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开输入视频：{args.in_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0  # 有些视频读不到 fps，给个合理默认值

    # 计算目标分辨率
    if args.width > 0 or args.height > 0:
        if args.keep_aspect:
            # 只给一边：按比例算另一边；两边都给：按较小缩放比适配
            if args.width > 0 and args.height == 0:
                out_w = args.width
                out_h = round(in_h * (out_w / in_w))
            elif args.height > 0 and args.width == 0:
                out_h = args.height
                out_w = round(in_w * (out_h / in_h))
            else:
                s = min(args.width / in_w, args.height / in_h)
                out_w = round(in_w * s)
                out_h = round(in_h * s)
        else:
            out_w = args.width if args.width > 0 else in_w
            out_h = args.height if args.height > 0 else in_h
    elif args.scale > 0:
        out_w = max(2, int(round(in_w * args.scale)))
        out_h = max(2, int(round(in_h * args.scale)))
    else:
        # 默认：降到一半
        out_w = max(2, in_w // 2)
        out_h = max(2, in_h // 2)

    # 确保偶数尺寸（很多编码器更喜欢偶数）
    out_w -= out_w % 2
    out_h -= out_h % 2

    interp_map = {
        "area": cv2.INTER_AREA,     # 缩小推荐
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC
    }
    interp = interp_map[args.interp]

    writer, used_codec = open_writer(args.out_path, fps, out_w, out_h)

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resized = cv2.resize(frame, (out_w, out_h), interpolation=interp)
        writer.write(resized)
        frame_count += 1

    cap.release()
    writer.release()

    print("Done.")
    print(f"Input : {args.in_path} ({in_w}x{in_h}, fps={fps:.3f})")
    print(f"Output: {args.out_path} ({out_w}x{out_h}, codec={used_codec}, frames={frame_count})")


if __name__ == "__main__":
    main()

# python src.utils.downscale.py --in /home/v-weicaiyan/ds/DATA/ego4d/v2/full_scale/e7742d04-2dc5-4046-9d0c-d45c6fe74d25.mp4 --out tmp/output.mp4 --width 480 --keep_aspect
