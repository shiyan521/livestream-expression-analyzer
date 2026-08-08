#!/usr/bin/env python3
"""
ASR后端抽象接口模块。

定义所有ASR后端必须实现的统一接口，使得上层分析逻辑
可以透明地切换不同的ASR实现（Whisper / 百炼 / 逐字稿导入等）。

子类必须实现 transcribe() 方法。recognize() 和 get_audio_duration()
提供了默认实现，子类可按需覆盖。
"""
import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class ASRBackend(ABC):
    """ASR后端抽象接口。

    所有ASR后端实现都必须继承此类并实现 transcribe 方法。
    返回的数据格式统一为 ASR diarization JSON 格式。

    可选覆盖方法:
        - recognize(): 转写并保存到文件（默认调用 transcribe + json.dump）
        - get_audio_duration(): 获取音频时长（默认使用 librosa）
    """

    @abstractmethod
    def transcribe(self, audio_path: str, speaker_count: int = 2) -> Dict[str, Any]:
        """转写音频并做说话人分离。

        Args:
            audio_path: 音频文件路径
            speaker_count: 说话人数量

        Returns:
            ASR diarization JSON格式：
            {
                "transcripts": [{
                    "sentences": [
                        {
                            "text": "...",
                            "speaker_id": 0,
                            "begin_time": 1234,
                            "end_time": 5678,
                            "words": []
                        }
                    ]
                }],
                "properties": {
                    "original_duration_in_milliseconds": 3600000
                }
            }
        """
        raise NotImplementedError

    def recognize(
        self,
        audio_path: str,
        speaker_count: int,
        output_json: str,
        timeout: int = 300,
    ) -> str:
        """转写音频并保存结果到 JSON 文件。

        默认实现调用 transcribe() 并将结果写入文件。
        子类可覆盖此方法以直接输出到文件（避免中间 dict 开销）。

        Args:
            audio_path: 音频文件路径。
            speaker_count: 预期说话人数量。
            output_json: 输出 JSON 文件路径。
            timeout: 超时时间（秒），子类可选实现。

        Returns:
            输出 JSON 文件路径。
        """
        result = self.transcribe(audio_path, speaker_count)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return output_json

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）。

        默认实现使用 librosa，子类可覆盖以使用更高效的方式。

        Args:
            audio_path: 音频文件路径。

        Returns:
            音频时长（秒）。
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr
        del y, sr
        return duration
