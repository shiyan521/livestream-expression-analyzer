#!/usr/bin/env python3
"""
逐字稿导入ASR后端（零API，最准确）。

接收用户提供带说话人标注的逐字稿文件，转换为 ASR diarization JSON 格式。
不需要调用任何ASR API，准确度取决于用户提供的逐字稿质量。

支持多种真实世界的逐字稿格式（fmt 参数或自动检测）：
  1. standard: 每行 `说话人(HH:MM:SS): 文字内容`（单行标准格式）
  2. meeting:  腾讯会议/直播平台段落式格式：`姓名 时间` 单独一行，
               下一行（或连续多行）为说话内容，直到下一条姓名行
  3. feishu:   飞书妙记导出格式：`[HH:MM:SS] 说话人 内容`
               或 `[HH:MM:SS] 说话人` + 内容行
  4. json:     `[{"speaker": "名字", "time": "HH:MM:SS", "text": "内容"}, ...]`

说话人映射通过参数传入，将说话人名称映射为 speaker_id；
未提供映射时自动按名字顺序编号。

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

# 标准单行格式: 说话人(HH:MM:SS): 文字内容（时间支持 MM:SS 或 HH:MM:SS）
STANDARD_LINE_PATTERN = re.compile(
    r"^(.+?)\((\d{1,2}:\d{2}(?::\d{2})?)\)\s*[:：]\s*(.*)"
)

# 腾讯会议/直播段落式格式: 姓名 + 空白 + 时间 单独成行（MM:SS 或 HH:MM:SS）
NAME_TIME_PATTERN = re.compile(
    r"^([\u4e00-\u9fa5A-Za-z0-9_·. ]{1,24})\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)

# 飞书妙记格式: [HH:MM:SS] 说话人[:：] 内容（或 [HH:MM:SS] 说话人 + 下一行内容）
FEISHU_PATTERN = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)\s*(?:[:：]\s*(.*))?$")

# 元信息行（标题/日期等），段落式解析时跳过
META_PATTERN = re.compile(r"^(第?\d+期?|\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}|\d{4}-\d{1,2}-\d{1,2}|[\u4e00-\u9fa5]{0,10}转写|转写记录|原文|记录)")


def _time_to_ms(time_str: str) -> int:
    """将时间字符串转换为毫秒。

    Args:
        time_str: 时间字符串，如 "01:23:45" 或 "01:23" 或 "83.5"

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


def _fix_end_times(sentences: List[Dict[str, Any]]) -> None:
    """修正 end_time：每句的 end_time 设为下一句的 begin_time，最后一句 +1s。"""
    for i in range(len(sentences)):
        if i < len(sentences) - 1:
            sentences[i]["end_time"] = sentences[i + 1]["begin_time"]
        else:
            sentences[i]["end_time"] = sentences[i]["begin_time"] + 1000


def _resolve_speaker_id(name: str, speaker_map: Dict[str, int], auto_map: Dict[str, int]) -> int:
    """解析说话人名称到 speaker_id；speaker_map 为空时自动编号。"""
    if speaker_map:
        return speaker_map.get(name)
    if name not in auto_map:
        auto_map[name] = len(auto_map)
    return auto_map[name]


def _parse_standard_format(
    content: str,
    speaker_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """解析标准单行格式逐字稿。

    格式：每行 `说话人(HH:MM:SS): 文字内容`

    Args:
        content: 逐字稿文本内容
        speaker_map: 说话人名称到 speaker_id 的映射

    Returns:
        句子列表，每个元素包含 text, speaker_id, begin_time, end_time, words 字段
    """
    sentences: List[Dict[str, Any]] = []
    auto_map: Dict[str, int] = {}
    lines = content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = STANDARD_LINE_PATTERN.match(line)
        if not match:
            print(f"  [WARN] 无法解析行，跳过: {line[:80]}...")
            continue

        speaker_name = match.group(1).strip()
        time_str = match.group(2)
        text = match.group(3).strip()

        if not text:
            # 时间戳行但无文字内容，跳过（不影响时间线）
            continue

        speaker_id = _resolve_speaker_id(speaker_name, speaker_map, auto_map)
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

    _fix_end_times(sentences)
    return sentences


def _parse_meeting_paragraph_format(
    content: str,
    speaker_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """解析腾讯会议/直播平台段落式逐字稿。

    格式示例（真实导出）：
        施言   01:12
        Hello hello hello，可以听到吗？

        禀宪   01:16
        可以，我这边可以听到。
        第二行内容会追加到当前句子。

    特征：`姓名 + 空白 + 时间(MM:SS 或 HH:MM:SS)` 单独成行，
    后续所有非姓名行都属于该说话人的内容（跨行合并），
    直到下一条姓名时间行。文件头部的标题/日期等元信息行自动跳过。

    Args:
        content: 逐字稿文本内容
        speaker_map: 说话人名称到 speaker_id 的映射

    Returns:
        句子列表
    """
    sentences: List[Dict[str, Any]] = []
    auto_map: Dict[str, int] = {}
    current: Optional[Dict[str, Any]] = None
    lines = content.strip().split("\n")
    started = False  # 是否已遇到第一个说话人行

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = NAME_TIME_PATTERN.match(line)
        if match:
            # 日期+时间等元信息行（如 "2026年07月09日 16:15"）也会命中此模式，需过滤
            if META_PATTERN.match(line):
                current = None
                continue
            # 新说话人开始
            started = True
            speaker_name = match.group(1).strip()
            time_str = match.group(2)
            speaker_id = _resolve_speaker_id(speaker_name, speaker_map, auto_map)
            if speaker_id is None:
                print(f"  [WARN] 未知说话人 '{speaker_name}'，跳过该段")
                current = None
                continue
            current = {
                "begin_time": _time_to_ms(time_str),
                "end_time": _time_to_ms(time_str),
                "text": "",
                "speaker_id": speaker_id,
                "words": [],
            }
            sentences.append(current)
        elif current is not None:
            # 追加到当前说话人内容（跨行合并）
            current["text"] += line
        elif not started:
            # 文件头部的标题/日期等元信息行，静默跳过
            continue
        elif not META_PATTERN.match(line):
            # 无当前说话人且非元信息行：可能是格式异常，跳过并警告
            print(f"  [WARN] 无法定位说话人，跳过: {line[:80]}...")

    # 移除空内容句子
    sentences = [s for s in sentences if s["text"].strip()]
    _fix_end_times(sentences)
    return sentences


def _parse_feishu_format(
    content: str,
    speaker_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """解析飞书妙记导出格式逐字稿。

    格式示例：
        [00:00:00] 张三：大家好，欢迎来到直播间
        [00:00:05] 李四
        谢谢邀请，今天我们来聊 AI 教育。

    支持同一行内带内容，也支持内容在下一行（跨行合并）。

    Args:
        content: 逐字稿文本内容
        speaker_map: 说话人名称到 speaker_id 的映射

    Returns:
        句子列表
    """
    sentences: List[Dict[str, Any]] = []
    auto_map: Dict[str, int] = {}
    current: Optional[Dict[str, Any]] = None
    lines = content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            current = None
            continue

        match = FEISHU_PATTERN.match(line)
        if match:
            time_str = match.group(1)
            speaker_part = match.group(2).strip()
            inline_text = (match.group(3) or "").strip()

            # 说话人部分可能形如 "张三：内容" 或 "张三"（含冒号时内容在 group(3)）
            speaker_name = speaker_part.rstrip("：:").strip()
            speaker_id = _resolve_speaker_id(speaker_name, speaker_map, auto_map)
            if speaker_id is None:
                print(f"  [WARN] 未知说话人 '{speaker_name}'，跳过")
                current = None
                continue

            current = {
                "begin_time": _time_to_ms(time_str),
                "end_time": _time_to_ms(time_str),
                "text": inline_text,
                "speaker_id": speaker_id,
                "words": [],
            }
            sentences.append(current)
        elif current is not None:
            # 无时间戳行：内容追加到当前句子（跨行合并）
            current["text"] += line

    sentences = [s for s in sentences if s["text"].strip()]
    _fix_end_times(sentences)
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
    auto_map: Dict[str, int] = {}

    for item in data:
        speaker_name = item.get("speaker", "")
        time_str = item.get("time", "00:00:00")
        text = item.get("text", "").strip()

        if not text:
            continue

        speaker_id = _resolve_speaker_id(speaker_name, speaker_map, auto_map)
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

    _fix_end_times(sentences)
    return sentences


FORMAT_PARSERS = {
    "standard": _parse_standard_format,
    "meeting": _parse_meeting_paragraph_format,
    "feishu": _parse_feishu_format,
    "json": _parse_json_format,
}


class TranscriptImportBackend(ASRBackend):
    """逐字稿导入ASR后端。

    从用户提供的带说话人标注的逐字稿文件导入数据，
    转换为统一的 ASR diarization JSON 格式。
    零API调用，准确度取决于逐字稿质量。

    Attributes:
        speaker_map: 说话人名称到 speaker_id 的映射
        format: 逐字稿格式，'standard'/'meeting'/'feishu'/'json' 或 'auto'（自动检测）
    """

    def __init__(
        self,
        speaker_map: Optional[Dict[str, int]] = None,
        fmt: str = "auto",
    ) -> None:
        """初始化逐字稿导入后端。

        Args:
            speaker_map: 说话人名称到 speaker_id 的映射。
                         例如 {"Speaker A": 0, "Speaker B": 1}；
                         未提供时按名字出现顺序自动编号。
            fmt: 逐字稿格式，'standard'（单行标准）、'meeting'（腾讯会议段落式）、
                 'feishu'（飞书妙记）、'json' 或 'auto'（自动检测）
        """
        self.speaker_map = speaker_map or {}
        self.fmt = fmt

    def _detect_format(self, content: str) -> str:
        """自动检测逐字稿格式。

        通过统计前 60 行中各格式模式的命中数，选择命中最多的格式。

        Args:
            content: 逐字稿文本内容

        Returns:
            'json' / 'standard' / 'meeting' / 'feishu' 之一
        """
        stripped = content.strip()
        # 飞书妙记格式以 [HH:MM:SS] 开头，需在 JSON 判断之前排除
        if re.match(r"^\[\d{1,2}:\d{2}", stripped):
            return "feishu"
        # JSON 格式以 [ 或 { 开头
        if stripped.startswith("[") or stripped.startswith("{"):
            return "json"

        lines = [l.strip() for l in stripped.split("\n") if l.strip()][:60]
        scores = {"standard": 0, "meeting": 0, "feishu": 0}
        for line in lines:
            if STANDARD_LINE_PATTERN.match(line):
                scores["standard"] += 2
            elif FEISHU_PATTERN.match(line):
                scores["feishu"] += 2
            elif NAME_TIME_PATTERN.match(line):
                scores["meeting"] += 1

        best_fmt = max(scores, key=scores.get)
        if scores[best_fmt] == 0:
            # 都没有命中：兜底尝试段落式（可能文件头有元信息行）
            return "meeting"
        return best_fmt

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
        parser = FORMAT_PARSERS.get(fmt)
        if parser is None:
            print(f"  [WARN] 未知格式 '{fmt}'，使用 meeting 解析器兜底")
            parser = _parse_meeting_paragraph_format
        sentences = parser(content, self.speaker_map)

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
