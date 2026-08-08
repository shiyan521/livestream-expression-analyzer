#!/usr/bin/env python3
"""
ASR后端工厂模块。

根据名称创建对应的ASR后端实例，支持 whisper / bailian / transcript 三种后端。
"""
from typing import Any

from .base import ASRBackend


def get_backend(backend_name: str, **kwargs: Any) -> ASRBackend:
    """根据名称获取ASR后端。

    Args:
        backend_name: 后端名称，支持以下值：
            - 'whisper': Whisper本地后端（默认，免费）
            - 'bailian': 阿里云百炼后端（付费，需安装 bl CLI）
            - 'transcript': 逐字稿导入后端（零API，最准确）
        **kwargs: 后端特定参数，将传递给后端构造函数。
            whisper 后端参数:
                model_name, language, use_pyannote, hf_token, pyannote_model
            bailian 后端参数:
                bl_path, timeout, sample_rate
            transcript 后端参数:
                speaker_map, fmt

    Returns:
        ASRBackend 实例

    Raises:
        ValueError: 当 backend_name 不支持时
    """
    backend_name = backend_name.lower().strip()

    if backend_name == "whisper":
        from .whisper_backend import WhisperBackend
        return WhisperBackend(**kwargs)

    elif backend_name == "bailian":
        from .bailian_backend import BailianBackend
        return BailianBackend(**kwargs)

    elif backend_name in ("transcript", "import"):
        from .transcript_import import TranscriptImportBackend
        return TranscriptImportBackend(**kwargs)

    else:
        raise ValueError(
            f"不支持的ASR后端: '{backend_name}'。"
            f"支持的后端: 'whisper', 'bailian', 'transcript'"
        )
