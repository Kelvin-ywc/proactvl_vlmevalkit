import debugpy
from torch.utils.data import Dataset
import os
import json
import math
from PIL import Image
import torch
from typing import Optional, List, Dict, Any, Union, Iterable, Tuple
import re
from qwen_omni_utils.v2_5.vision_process import extract_vision_info, fetch_image, fetch_video, process_vision_info
from qwen_omni_utils.v2_5.audio_process import process_audio_info
from dataclasses import dataclass
import random
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download
from .logger import Logger
import numpy as np
from io import BytesIO
import base64


logger = Logger(__name__).get_logger()

def debug(port: int)-> None:
    try:
        debugpy.listen(("localhost", port))
        logger.info(f"🕛 Waiting for debugger to connect on port {port}...")
        debugpy.wait_for_client()
    except Exception as e:
        logger.info(f"Error starting debugpy: {e}")

def sizeof(numel: int, dtype: torch.dtype) -> int:
    """估算张量字节大小（不考虑稀疏/共享存储）。"""
    # 常见 dtype 大小（字节）
    sizes = {
        torch.float64: 8, torch.int64: 8, torch.long: 8,
        torch.bfloat16: 2, torch.float16: 2, torch.half: 2,
        torch.float32: 4, torch.int32: 4,
        torch.int16: 2, torch.short: 2,
        torch.uint8: 1, torch.int8: 1, torch.bool: 1,
    }
    return numel * sizes.get(dtype, 4)  # 未知 dtype 默认按 4B 估

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def iter_named_tensors(model: torch.nn.Module, include_buffers: bool = True) -> Iterable[Tuple[str, torch.Tensor, str]]:
    # 参数
    for name, p in model.named_parameters(recurse=True):
        yield name, p, "param"
    # 可选：buffer
    if include_buffers:
        for name, b in model.named_buffers(recurse=True):
            yield name, b, "buffer"

def print_model_param_devices(model: torch.nn.Module, include_buffers: bool = False, max_rows: int = None):
    """
    打印模型中每个参数（以及可选的 buffer）的：名字、形状、dtype、device、numel、大小。
    同时给出按设备的汇总。
    """
    rows = []
    per_device_bytes = {}
    per_device_numel = {}

    for name, t, kind in iter_named_tensors(model, include_buffers=include_buffers):
        # 处理可能的 meta 张量
        try:
            device = str(t.device)
        except Exception:
            device = "unknown"

        shape = tuple(t.shape)
        dtype = t.dtype
        numel = t.numel() if t.is_meta is False else 0
        nbytes = sizeof(numel, dtype) if numel > 0 else 0

        rows.append((name, shape, str(dtype).replace("torch.", ""), device, kind, numel, human_bytes(nbytes)))
        if "meta" not in device:
            per_device_bytes[device] = per_device_bytes.get(device, 0) + nbytes
            per_device_numel[device] = per_device_numel.get(device, 0) + numel

    # 打印表头
    header = f"{'name':<64} {'shape':<24} {'dtype':<10} {'device':<10} {'type':<8} {'#elms':>12} {'size':>8}"
    print(header)
    print("-" * len(header))

    # 打印明细
    shown = 0
    for name, shape, dtype, device, kind, numel, size_str in rows:
        if max_rows is not None and shown >= max_rows:
            break
        shape_str = "()" if len(shape) == 0 else str(shape)
        if len(shape_str) > 24:
            shape_str = shape_str[:21] + "..."
        name_disp = name if len(name) <= 64 else name[:61] + "..."
        print(f"{name_disp:<64} {shape_str:<24} {dtype:<10} {device:<10} {kind:<8} {numel:>12} {size_str:>8}")
        shown += 1

    # 若有截断提醒
    if max_rows is not None and shown < len(rows):
        print(f"... ({len(rows) - shown} more rows)")

    # 设备汇总
    if per_device_bytes:
        print("\nPer-device summary:")
        for dev, bytes_ in per_device_bytes.items():
            numel = per_device_numel.get(dev, 0)
            print(f"  {dev:<10} params={numel:>12}  size={human_bytes(bytes_):>8}")

    # 总计
    total_bytes = sum(per_device_bytes.values())
    total_numel = sum(per_device_numel.values())
    print(f"\nTotal (non-meta): params={total_numel}  size={human_bytes(total_bytes)}")

# ---------------------------
# 用法示例：
# print_model_param_devices(model, include_buffers=True)
# print_model_param_devices(model, include_buffers=False, max_rows=200)

def replace_segment_with_chunk(texts, replace_list):
    """
    对 texts 中的每个字符串：
    将第 i 个 <|VIDEO|> 替换为 replace_list[i]。
    要求每个 text 里的 <|VIDEO|> 数量 == len(replace_list)
    """
    def replace_in_text(text):
        count = 0  # 每个 text 独立计数
        def repl(match):
            nonlocal count
            if count >= len(replace_list):
                raise ValueError("某个 text 中 <|VIDEO|> 数量超过 replace_list 长度")
            replacement = replace_list[count]
            count += 1
            return replacement

        new_text = re.sub(r"<\|vision_bos\|><\|VIDEO\|><\|vision_eos\|>", repl, text)

        if count != len(replace_list):
            raise ValueError("某个 text 中 <|VIDEO|> 数量与 replace_list 长度不一致")

        return new_text

    return [replace_in_text(text) for text in texts]

def prune_cache_span(cache, start: int, end: int):
    """
    在 DynamicCache 上“就地”删除 [start, end) 这段序列（所有层）。
    仅做张量切片拼接 + seen_tokens 修正。调用方需自行确保与文本/位置对齐！
    """

    num_layers = len(cache.layers)
    for i in range(num_layers):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        # 常见 layout 1: [B, H, T, D]
        # FIXME, hard coding, for qwen 2.5 omni, cache shape: [B, H, Seq_len, D]

        k_new = torch.cat([k[:, :, :start, :], k[:, :, end:, :]], dim=2)
        v_new = torch.cat([v[:, :, :start, :], v[:, :, end:, :]], dim=2)

        cache.layers[i].keys = k_new
        cache.layers[i].values = v_new
    return cache

def prune_cache_span_v1(cache, start: int, end: int):
    """
    在 DynamicCache 上“就地”删除 [start, end) 这段序列（所有层）。
    仅做张量切片拼接 + seen_tokens 修正。调用方需自行确保与文本/位置对齐！
    """
    assert hasattr(cache, "key_cache") and hasattr(cache, "value_cache"), "Not a DynamicCache-like object"
    L = int(getattr(cache, "seen_tokens", None) or 0)
    assert 0 <= start < end <= L, f"bad range [{start},{end}) for length {L}"
    drop = end - start

    num_layers = len(cache.key_cache)
    for i in range(num_layers):
        k = cache.key_cache[i]
        v = cache.value_cache[i]
        # 常见 layout 1: [B, H, T, D]
        # FIXME, hard coding, for qwen 2.5 omni, cache shape: [B, H, Seq_len, D]

        k_new = torch.cat([k[:, :, :start, :], k[:, :, end:, :]], dim=2)
        v_new = torch.cat([v[:, :, :start, :], v[:, :, end:, :]], dim=2)

        cache.key_cache[i] = k_new
        cache.value_cache[i] = v_new

    # 修正长度
    new_len = L - drop
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = new_len
    return cache

def _get(s: Union[dict, Any], k: str, default=None):
    return s.get(k, default) if isinstance(s, dict) else getattr(s, k, default)

def _squeeze_batch1(x: torch.Tensor) -> torch.Tensor:
    # 把 [1, L] / [1, ...] 变成 [L] / [...]
    return x.squeeze(0) if isinstance(x, torch.Tensor) and x.dim() > 0 and x.size(0) == 1 else x

def to_model_dtype(batch, model_dtype=torch.bfloat16):
    out = {}
    for k, v in batch.items():
        if v.dtype in (torch.float32, torch.float64, torch.float16):
            out[k] = v.to(dtype=model_dtype)
        else:
            out[k] = v
    return out

# 采样工具：temperature + nucleus (top-p)
def sample_from_logits(last_logits, temperature=1.0, top_p=1.0):
    # last_logits: (1, vocab)
    l = last_logits / max(float(temperature), 1e-6)
    probs = torch.softmax(l, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # 保留前缀概率和 <= top_p 的最小集合
        keep_mask_sorted = cum <= top_p
        keep_mask_sorted[:, 0] = True  # 至少保留一个
        # 还原到原索引位置
        keep_mask = torch.zeros_like(probs, dtype=torch.bool)
        keep_mask.scatter_(1, sorted_idx, keep_mask_sorted)
        probs = torch.where(keep_mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
    return next_token

def whether_finish(text: str) -> bool:
    """
    判断文本是否“结束”。
    目前的启发式标准是：文本以句号/问号/感叹号结尾。
    """
    text = text.strip()
    if len(text) == 0:
        return False
    return text[-1] in {'.', '!', '?'}

def _maybe_hf(repo_or_dir: str, filename: str, revision: Optional[str] = None, subfolder: Optional[str] = None) -> str:
    """
    如果是本地目录，返回拼好的本地路径；
    如果是 HF repo_id，则用 hf_hub_download 下载（支持 subfolder）。
    filename：相对 subfolder 的文件名，比如 "state_proj_config.json"
    """
    if _is_local_dir(repo_or_dir):
        base = repo_or_dir
        if subfolder:
            base = os.path.join(base, subfolder)
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local file not found: {path}")
        return path
    # HF 仓库
    return hf_hub_download(
        repo_id=repo_or_dir,
        filename=os.path.join(subfolder, filename) if subfolder else filename,
        revision=revision,
        local_dir=None
    )

def _is_local_dir(path: str) -> bool:
    return os.path.isdir(path)


def _split_words(text: str) -> List[str]:
    if text is None:
        return []
    cleaned = text.replace('\n', ' ').replace('<|im_end|>', ' ').strip()
    parts = [w for w in cleaned.split() if len(w) > 0]
    if len(parts) == 1 and any('\u4e00' <= ch <= '\u9fff' for ch in parts[0]):
        return [ch for ch in parts[0] if ch.strip()]
    return parts

def _ends_with_punct(words: List[str]) -> bool:
    punctuation_chars = set("。！？!?;；…,.，、:：")
    if not words:
        return False
    last = words[-1]
    return len(last) > 0 and last[-1] in punctuation_chars

def split_text_into_segments(text, n_segments):
    # 将 text 按空格拆分成 n_segments 段，尽量均匀分配单词, 如果segments段数小于n_segments, 则只取实际段数,空格需要保留
    words = text.split()
    # mask = []
    total_words = len(words)
    if total_words == 0:
        return [""] * n_segments
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

def frame_to_base64(frame, fmt="JPEG"):
    """
    frame: [3, H, W]，可以是 torch.Tensor 或 np.ndarray
    fmt: "JPEG" 或 "PNG"
    return: 不带 data: 前缀的 base64 字符串
    """

    # 1. 转成 numpy
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    # frame 现在是 (3, H, W)
    assert frame.shape[0] == 3, f"expect C=3, got {frame.shape}"

    # 2. 如果是 0~1 的浮点数，先变成 0~255 的 uint8
    if frame.dtype != np.uint8 and frame.min() >= 0.0 and frame.max() <= 1.0:
        frame = np.clip(frame, 0, 1) * 255
        frame = frame.astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # 3. 维度从 (C, H, W) -> (H, W, C)
    frame = np.transpose(frame, (1, 2, 0))

    # 4. 转成 PIL Image
    img = Image.fromarray(frame)

    # 5. 写入内存 buffer 并转 base64
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    img_bytes = buffer.getvalue()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64