#!/usr/bin/env python3
"""
逐字稿导入ASR后端（零API，最准确）。

接收用户提供带说话人标注的逐字稿文件，转换为 ASR diarization JSON 格式。
不需要调用任何ASR API，准确度取决于用户提供的逐字稿质量。

支持两种格式：
  1. 腾讯会议格式：每行 `说话人(HH:MM:SS): 文字内容`
  2. JSON格式：`[{"speaker": "名字", "time": "HH:MM:SS", "text": "内容"}, ...]`

说话人映射通过参数传入，将说话人名称映射为 speaker_id。

用法示例:
  backend = TranscriptImportBackend(
      speaker_map={"Speaker A": 0, "Speaker B": 1}
  )
  result = backend.transcribe("/path/to/transcript.txt", speaker_count=3)
"""
import json
import re
from typing import Any, Dict, List, Optional

from .base import ASRBackend

# 腾讯会议逐字稿行匹配正则: 说话人(HH:MM:SS): 文字内容
TENCENT_LINE_PATTERN = re.compile(r"^(.+?)\((\d{2}:\d{2}:\d{2})\):\s*(.*)")


def _time_to_ms(time_str: str) -> int:
    """将 HH:MM:SS 时间字符串转换为毫秒。

    Args:
        time_str: 时间字符串，如 "01:23:45"

    Returns:
        毫秒数
    """
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60000 + int(s) * 1000
    else:
        return int(float(time_str) * 1000)


def _parse_tencent_format(
    content: str,
    speaker_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """解析腾讯会议格式逐字稿。

    格式：每行 `说话人(HH:MM:SS): 文字内容`

    Args:
        content: 逐字稿文本内容
        speaker_map: 说话人名称到 speaker_id 的映射

    Returns:
        句子列表，每个元素包含 text, speaker_id, begin_time, end_time, words 字段
    """
    sentences: List[Dict[str, Any]] = []
    lines = content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = TENCENT_LINE_PATTERN.match(line)
        if not match:
            print(f"  [WARN] 无法解析行，跳过: {line[:80]}...")
            continue

        speaker_name = match.group(1).strip()
        time_str = match.group(2)
        text = match.group(3).strip()

        if not text:
            # 时间戳行但无文字内容，跳过（不影响时间线）
            continue

        speaker_id = speaker_map.get(speaker_name)
        if speaker_id is None:
            print(f"  [WARN] 未知说话人 '{speaker_name}'，跳过: {line[:80]}...")
            continue

        begin_time = _time_to_ms(time_str)
        sentences.append({
            "begin_time": begin_time,
            "end_time": begin_time,  # 占位，稍后修正
            "text": text,
            "speaker_id": speaker_id,
            "words": [],
        })

    # 修正 end_time：每句的 end_time 设为下一句的 begin_time
    # 最后一句的 end_time 设为 begin_time + 1000ms（保守估计至少1秒）
    for i in range(len(sentences)):
        if i < len(sentences) - 1:
            sentences[i]["end_time"] = sentences[i + 1]["begin_time"]
        else:
            sentences[i]["end_time"] = sentences[i]["begin_time"] + 1000

    return sentences


def _parse_json_format(
    content: str,
    speaker_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """解析JSON格式逐字稿。

    格式：`[{"speaker": "名字", "time": "HH:MM:SS", "text": "内容"}, ...]`

    Args:
        content: JSON 文本内容
        speaker_map: 说话人名称到 speaker_id 的映射

    Returns:
        句子列表
    """
    data = json.loads(content)
    sentences: List[Dict[str, Any]] = []

    for item in data:
        speaker_name = item.get("speaker", "")
        time_str = item.get("time", "00:00:00")
        text = item.get("text", "").strip()

        if not text:
            continue

        speaker_id = speaker_map.get(speaker_name)
        if speaker_id is None:
            print(f"  [WARN] 未知说话人 '{speaker_name}'，跳过")
            continue

        begin_time = _time_to_ms(time_str)
        sentences.append({
            "begin_time": begin_time,
            "end_time": begin_time,  # 占位
            "text": text,
            "speaker_id": speaker_id,
            "words": [],
        })

    # 修正 end_time
    for i in range(len(sentences)):
        if i < len(sentences) - 1:
            sentences[i]["end_time"] = sentences[i + 1]["begin_time"]
        else:
            sentences[i]["end_time"] = sentences[i]["begin_time"] + 1000

    return sentences


class TranscriptImportBackend(ASRBackend):
    """逐字稿导入ASR后端。

    从用户提供的带说话人标注的逐字稿文件导入数据，
    转换为统一的 ASR diarization JSON 格式。
    零API调用，准确度取决于逐字稿质量。

    Attributes:
        speaker_map: 说话人名称到 speaker_id 的映射
        format: 逐字稿格式，'tencent' 或 'json' 或 'auto'（自动检测）
    """

    def __init__(
        self,
        speaker_map: Optional[Dict[str, int]] = None,
        fmt: str = "auto",
    ) -> None:
        """初始化逐字稿导入后端。

        Args:
            speaker_map: 说话人名称到 speaker_id 的映射。
                         例如 {"Speaker A": 0, "Speaker B": 1}
            fmt: 逐字稿格式，'tencent'（腾讯会议格式）、'json' 或 'auto'（自动检测）
        """
        self.speaker_map = speaker_map or {}
        self.fmt = fmt

    def _detect_format(self, content: str) -> str:
        """自动检测逐字稿格式。

        Args:
            content: 逐字稿文本内容

        Returns:
            'tencent' 或 'json'
        """
        stripped = content.strip()
        # JSON 格式以 [ 或 { 开头
        if stripped.startswith("[") or stripped.startswith("{"):
            return "json"
        # 否则默认为腾讯会议格式
        return "tencent"

    def transcribe(self, audio_path: str, speaker_count: int = 2) -> Dict[str, Any]:
        """从逐字稿文件导入数据。

        注意：此方法中 audio_path 参数实际指向逐字稿文件路径，
        而非音频文件路径。

        Args:
            audio_path: 逐字稿文件路径（txt 或 json）
            speaker_count: 说话人数量（仅用于参考，实际说话人由逐字稿内容决定）

        Returns:
            ASR diarization JSON格式
        """
        print(f"[TranscriptImportBackend] 导入逐字稿: {audio_path}")

        with open(audio_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 确定格式
        fmt = self.fmt
        if fmt == "auto":
            fmt = self._detect_format(content)
        print(f"  检测到格式: {fmt}")

        # 解析
        if fmt == "json":
            sentences = _parse_json_format(content, self.speaker_map)
        else:
            sentences = _parse_tencent_format(content, self.speaker_map)

        print(f"  解析完成: {len(sentences)} 句")

        # 统计各说话人句数
        speaker_counts: Dict[int, int] = {}
        for s in sentences:
            sid = s["speaker_id"]
            speaker_counts[sid] = speaker_counts.get(sid, 0) + 1
        print(f"  各说话人句数: {speaker_counts}")

        # 计算总时长
        if sentences:
            max_end_time = sentences[-1]["end_time"]
        else:
            max_end_time = 0
        original_duration = max_end_time + 1000  # 加1秒余量

        # 构建输出结构
        return {
            "transcripts": [{"sentences": sentences}],
            "properties": {
                "original_duration_in_milliseconds": original_duration,
            },
        }
