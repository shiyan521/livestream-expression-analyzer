"""
互动节奏分析模块。

包含以下功能:
- 轮次构建（合并相邻同说话人片段）
- 回应时间统计
- 打断检测（重叠片段）
- 重叠率计算
- 主导度分析（各说话人发言占比、轮次统计）
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from analyzer.utils import fmt_time, safe_float


def compute_interaction(
    sentences: List[Dict[str, Any]],
    speaker_count: int,
) -> Dict[str, Any]:
    """轮次/回应时间/打断/主导比分析。

    Args:
        sentences: ASR 句子列表，每个句子包含 start, end, speaker 字段。
        speaker_count: 说话人数量。

    Returns:
        互动分析结果字典，包含:
            total_turns: 总轮次数
            dominance: 各说话人主导度统计
            overlap_ratio: 重叠率
            response_times: 回应时间统计
            interruptions: 打断事件列表
            transition_distribution: 转换时间分布
    """
    print("[7/9] 互动节奏分析...")
    if speaker_count < 2:
        return {"note": "独白场，无互动数据"}

    # 构建 turns
    turns: List[Dict[str, Any]] = []
    current: Dict[str, Any] = None
    for s in sentences:
        if current is None:
            current = {"speaker": s["speaker"], "start": s["start"], "end": s["end"]}
        elif s["speaker"] == current["speaker"] and s["start"] - current["end"] <= 5.0:
            current["end"] = s["end"]
        else:
            turns.append(current)
            current = {"speaker": s["speaker"], "start": s["start"], "end": s["end"]}
    if current:
        turns.append(current)

    # 各说话人统计
    speaker_time: Dict[int, float] = defaultdict(float)
    speaker_turns: Dict[int, int] = defaultdict(int)
    speaker_turn_lens: Dict[int, List[float]] = defaultdict(list)
    for t in turns:
        dur = t["end"] - t["start"]
        speaker_time[t["speaker"]] += dur
        speaker_turns[t["speaker"]] += 1
        speaker_turn_lens[t["speaker"]].append(dur)

    total_time = sum(speaker_time.values())

    # 回应时间
    response_times: Dict[int, List[float]] = defaultdict(list)
    transitions: List[float] = []
    for i in range(1, len(turns)):
        prev, curr = turns[i - 1], turns[i]
        if prev["speaker"] != curr["speaker"]:
            gap = curr["start"] - prev["end"]
            response_times[curr["speaker"]].append(gap)
            transitions.append(gap)

    overlap_count = sum(1 for g in transitions if g < 0.5)
    overlap_ratio = safe_float(overlap_count / len(transitions)) if transitions else 0

    # 打断检测
    interruptions: List[Dict[str, Any]] = []
    for i in range(1, len(turns)):
        prev, curr = turns[i - 1], turns[i]
        if prev["speaker"] != curr["speaker"]:
            gap = curr["start"] - prev["end"]
            if gap < 0:
                interruptions.append({
                    "time": fmt_time(curr["start"]),
                    "interrupter": int(curr["speaker"]),
                    "overlap_sec": safe_float(-gap),
                })

    # 主导度
    dominance: Dict[str, Dict[str, Any]] = {}
    for sp in range(speaker_count):
        dominance[str(sp)] = {
            "speaking_ratio": safe_float(speaker_time[sp] / total_time) if total_time > 0 else 0,
            "turns": speaker_turns[sp],
            "avg_turn_sec": safe_float(np.mean(speaker_turn_lens[sp])) if speaker_turn_lens[sp] else 0,
            "median_turn_sec": safe_float(np.median(speaker_turn_lens[sp])) if speaker_turn_lens[sp] else 0,
            "max_turn_sec": safe_float(max(speaker_turn_lens[sp])) if speaker_turn_lens[sp] else 0,
        }

    # 回应时间统计
    resp_stats: Dict[str, Dict[str, Any]] = {}
    for sp in range(speaker_count):
        rt = response_times[sp]
        if rt:
            resp_stats[str(sp)] = {
                "mean_sec": safe_float(np.mean(rt)),
                "median_sec": safe_float(np.median(rt)),
                "count": len(rt),
            }

    print(
        f"  轮次: {len(turns)}, 重叠率: {overlap_ratio:.1%}, "
        f"打断: {len(interruptions)}"
    )

    return {
        "total_turns": len(turns),
        "dominance": dominance,
        "overlap_ratio": overlap_ratio,
        "response_times": resp_stats,
        "interruptions": interruptions[:20],
        "transition_distribution": {
            "overlap_lt05s": overlap_count,
            "quick_05_2s": sum(1 for g in transitions if 0.5 <= g < 2.0),
            "normal_2_5s": sum(1 for g in transitions if 2.0 <= g < 5.0),
            "long_gt5s": sum(1 for g in transitions if g >= 5.0),
        },
    }
