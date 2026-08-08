"""
asr 模块 - 可插拔ASR后端。

提供统一的ASR后端接口和三种实现：
  - whisper: Whisper本地后端（默认，免费）
  - bailian: 阿里云百炼后端（付费，需 bl CLI）
  - transcript: 逐字稿导入后端（零API，最准确）

用法示例:
  from asr import get_backend

  # 使用Whisper后端
  backend = get_backend("whisper", model_name="medium")
  result = backend.transcribe("/path/to/audio.mp3", speaker_count=2)

  # 使用百炼后端
  backend = get_backend("bailian")
  result = backend.transcribe("/path/to/audio.mp3", speaker_count=2)

  # 使用逐字稿导入后端
  backend = get_backend("transcript", speaker_map={"Speaker A": 0, "Speaker B": 1})
  result = backend.transcribe("/path/to/transcript.txt", speaker_count=2)
"""

from .base import ASRBackend
from .bl_backend import BLASRBackend
from .factory import get_backend

__all__ = ["ASRBackend", "BLASRBackend", "get_backend"]
