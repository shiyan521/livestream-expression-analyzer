#!/usr/bin/env python3
"""
Whisper本地ASR后端（默认，免费）。

使用 openai-whisper 做语音转写，可选使用 pyannote.audio 做说话人分离。
如果没有安装 pyannote.audio，则所有句子标记为 speaker_id=0。

安装依赖:
  # 基础依赖（仅转写，无说话人分离）
  pip install openai-whisper

  # 可选：说话人分离（需要 pyannote.audio）
  pip install pyannote.audio
  # 注意：pyannote.audio 需要在 HuggingFace 上接受模型许可协议并配置 token
  # 设置环境变量: export HF_TOKEN=your_huggingface_token
"""
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from .base import ASRBackend


class WhisperBackend(ASRBackend):
    """Whisper本地ASR后端。

    使用 openai-whisper 进行语音转写，可选使用 pyannote.audio 进行说话人分离。
    如果没有 pyannote.audio，所有句子统一标记为 speaker_id=0。

    Attributes:
        model_name: Whisper模型名称（如 'base', 'small', 'medium', 'large'）
        language: 语言代码（如 'zh'），None 表示自动检测
        use_pyannote: 是否尝试使用 pyannote.audio 做说话人分离
        hf_token: HuggingFace token（pyannote需要）
        pyannote_model: pyannote 模型名称
    """

    def __init__(
        self,
        model_name: str = "medium",
        language: Optional[str] = "zh",
        use_pyannote: bool = True,
        hf_token: Optional[str] = None,
        pyannote_model: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        """初始化Whisper后端。

        Args:
            model_name: Whisper模型名称，默认 'medium'
            language: 语言代码，默认 'zh'（中文）
            use_pyannote: 是否尝试使用 pyannote.audio，默认 True
            hf_token: HuggingFace token，默认从环境变量 HF_TOKEN 读取
            pyannote_model: pyannote 模型名称
        """
        self.model_name = model_name
        self.language = language
        self.use_pyannote = use_pyannote
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.pyannote_model = pyannote_model
        self._whisper_model = None
        self._pyannote_pipeline = None

    def _get_whisper_model(self):
        """延迟加载Whisper模型。"""
        if self._whisper_model is None:
            import whisper
            print(f"[WhisperBackend] 加载 Whisper 模型: {self.model_name}...")
            self._whisper_model = whisper.load_model(self.model_name)
        return self._whisper_model

    def _get_pyannote_pipeline(self):
        """延迟加载pyannote说话人分离管线。

        Returns:
            pyannote Pipeline 实例，如果不可用则返回 None
        """
        if not self.use_pyannote:
            return None

        if self._pyannote_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                print(f"[WhisperBackend] 加载 pyannote 管线: {self.pyannote_model}...")
                self._pyannote_pipeline = Pipeline.from_pretrained(
                    self.pyannote_model,
                    use_auth_token=self.hf_token,
                )
            except ImportError:
                print("[WhisperBackend] pyannote.audio 未安装，跳过说话人分离。"
                      "所有句子将标记为 speaker_id=0。"
                      "安装方法: pip install pyannote.audio")
                self._pyannote_pipeline = False  # 标记为不可用
            except Exception as e:
                print(f"[WhisperBackend] pyannote 加载失败: {e}。"
                      "所有句子将标记为 speaker_id=0。")
                self._pyannote_pipeline = False

        if self._pyannote_pipeline is False:
            return None
        return self._pyannote_pipeline

    def _transcribe_with_whisper(self, audio_path: str) -> List[Dict[str, Any]]:
        """使用Whisper转写音频，返回句子列表。

        Args:
            audio_path: 音频文件路径

        Returns:
            句子列表，每个元素包含 text, begin_time, end_time 字段
        """
        model = self._get_whisper_model()
        print(f"[WhisperBackend] 转写中 (model={self.model_name}, language={self.language})...")

        result = model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
        )

        sentences: List[Dict[str, Any]] = []
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if not text:
                continue
            sentences.append({
                "text": text,
                "begin_time": int(segment["start"] * 1000),
                "end_time": int(segment["end"] * 1000),
                "words": segment.get("words", []),
            })

        print(f"[WhisperBackend] 转写完成，共 {len(sentences)} 句")
        return sentences

    def _diarize_with_pyannote(
        self,
        audio_path: str,
        speaker_count: int,
    ) -> List[Dict[str, Any]]:
        """使用pyannote做说话人分离。

        Args:
            audio_path: 音频文件路径
            speaker_count: 说话人数量

        Returns:
            说话人片段列表，每个元素包含 speaker_id, start, end 字段
        """
        pipeline = self._get_pyannote_pipeline()
        if pipeline is None:
            return []

        print(f"[WhisperBackend] 说话人分离中 (speakers={speaker_count})...")
        diarization = pipeline(
            audio_path,
            num_speakers=speaker_count,
        )

        segments: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # 从 "SPEAKER_00" 中提取数字
            speaker_id = int(speaker.split("_")[-1]) if speaker.startswith("SPEAKER_") else 0
            segments.append({
                "speaker_id": speaker_id,
                "start": turn.start,
                "end": turn.end,
            })

        print(f"[WhisperBackend] 说话人分离完成，共 {len(segments)} 个片段")
        return segments

    def _assign_speakers(
        self,
        sentences: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将说话人分离结果分配到Whisper句子。

        策略：取句子的中间时间点，找到包含该时间点的说话人片段。

        Args:
            sentences: Whisper转写的句子列表
            diarization_segments: pyannote的说话人片段列表

        Returns:
            带speaker_id的句子列表
        """
        if not diarization_segments:
            # 没有说话人分离结果，全部标记为0
            for s in sentences:
                s["speaker_id"] = 0
            return sentences

        for sent in sentences:
            # 取句子中间时间点（秒）
            mid_s = (sent["begin_time"] + sent["end_time"]) / 2 / 1000.0
            # 找到包含该时间点的说话人片段
            assigned = False
            for seg in diarization_segments:
                if seg["start"] <= mid_s <= seg["end"]:
                    sent["speaker_id"] = seg["speaker_id"]
                    assigned = True
                    break
            if not assigned:
                sent["speaker_id"] = 0

        return sentences

    def transcribe(self, audio_path: str, speaker_count: int = 2) -> Dict[str, Any]:
        """转写音频并做说话人分离。

        Args:
            audio_path: 音频文件路径
            speaker_count: 说话人数量

        Returns:
            ASR diarization JSON格式
        """
        # 1. Whisper转写
        sentences = self._transcribe_with_whisper(audio_path)

        # 2. 说话人分离（如果可用）
        diarization_segments = self._diarize_with_pyannote(audio_path, speaker_count)

        # 3. 分配说话人
        sentences = self._assign_speakers(sentences, diarization_segments)

        # 4. 计算总时长
        if sentences:
            max_end = max(s["end_time"] for s in sentences)
        else:
            max_end = 0

        # 确保所有句子都有 words 字段
        for s in sentences:
            if "words" not in s:
                s["words"] = []

        return {
            "transcripts": [{"sentences": sentences}],
            "properties": {
                "original_duration_in_milliseconds": max_end + 1000,
            },
        }
