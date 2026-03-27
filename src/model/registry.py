"""
Unified model registry for ProActVLM.
Usage:
    from src.model.registry import register_model, build_model
"""

from __future__ import annotations
from typing import Callable, Dict, Any, Optional, Type
import importlib
import inspect
import torch
from src.model.base import BaseModel

# --- Global Model Registry ---
_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def _norm(s: str) -> str:
    """小写 + 把 '-' '.' 替换成 '_' 并去掉多余空白。"""
    return (s or "").strip().lower().replace("-", "_").replace(".", "_")

def _tail(repo_id: str) -> str:
    """取 repo_id 最后一个段（'a/b/c' -> 'c'）。"""
    return repo_id.rsplit("/", 1)[-1]

def register(*keys: str):
    def deco(cls):
        for key in keys:
            k = _norm(key)
            if k in _MODEL_REGISTRY and _MODEL_REGISTRY[k] is not cls:
                raise ValueError(f"Key '{k}' already registered by {_MODEL_REGISTRY[k].__name__}")
            _MODEL_REGISTRY[k] = cls
        return cls
    return deco

def _get_model_class(model_name_or_path: str) -> Type[BaseModel]:
    """
    最简解析逻辑：
    1) 归一化整个 repo：exact 命中；
    2) 归一化尾段：exact 命中；
    3) 包含匹配：任一已注册 key 出现在归一化后的 repo或尾段中。
    """
    norm_full = _norm(model_name_or_path)
    norm_tail = _norm(_tail(model_name_or_path))
    print(model_name_or_path, norm_full, norm_tail)
    # 1) 精确匹配（全串）
    if norm_full in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[norm_full]
    # 2) 精确匹配（尾段）
    if norm_tail in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[norm_tail]
    # 3) 包含匹配（后注册优先，从后往前）
    for k in reversed(list(_MODEL_REGISTRY.keys())):
        if k in norm_tail or k in norm_full:
            return _MODEL_REGISTRY[k]

    available = ", ".join(sorted(_MODEL_REGISTRY.keys())) or "<empty>"
    raise ValueError(f"Unrecognized model_name_or_path='{model_name_or_path}'. "
                     f"Available keys: {available}")

def build_model(model_name_or_path: str, **kwargs) -> BaseModel:
    cls = _get_model_class(model_name_or_path)
    return cls(model_name_or_path=model_name_or_path, **kwargs)

