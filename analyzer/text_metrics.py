"""
文本指标模块。

从 ASR 逐字稿提取表达指标，包含:
- 口癖统计（核心 + 扩展）
- 语速计算
- 句长分布
- 停顿分布
- 碎片率
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from analyzer.utils import (
    fmt_time,
    load_filler_config,
    safe_float,
)


def compute_text_metrics(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    filler_config: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """从 ASR 逐字稿提取表达指标。

    Args:
        sentences: ASR 句子列表，每个句子包含 start, end, text, speaker 字段。
        target_speaker: 目标说话人 ID。
        filler_config: 口癖词库配置，若为 None 则从 config/fillers.yaml 加载。
            应包含 core_fillers、extended_fillers 键。

    Returns:
        文本指标字典，包含:
            meta: 元信息（总字数、发言时长、数据来源等）
            filler_words: 口癖统计
            speech_rate: 语速统计
            sentence_length: 句长分布
            pauses: 停顿分布
    """
    print("[5/9] 文本指标提取...")

    # 加载口癖词库
    if filler_config is None:
        filler_config = load_filler_config()
    filler_core = filler_config.get("core_fillers", [])
    filler_extended = filler_config.get("extended_fillers", [])

    target = [s for s in sentences if s["speaker"] == target_speaker]

    # --- 逐字稿口癖过滤检测 ---
    all_text = "".join(s["text"] for s in target)
    e_count = all_text.count("呃")
    n_count = all_text.count("嗯")
    transcript_warning = ""
    data_source = "asr"
    if len(all_text) > 500 and e_count == 0 and n_count == 0:
        transcript_warning = (
            'WARNING: 逐字稿中"呃""嗯"出现次数为0，'
            "可能是腾讯会议AI整理版，已系统性过滤轻声口癖。"
            "口癖密度数据将严重偏低，建议用 --asr-retry 跑ASR校准。"
        )
        data_source = "transcript_filtered"
        print(f"  ⚠ {transcript_warning}")
    elif len(all_text) > 500 and e_count + n_count < 5:
        transcript_warning = (
            f'WARNING: 逐字稿中"呃""嗯"仅出现{e_count + n_count}次（正常应为30+次），'
            "口癖数据可能被部分过滤。"
        )
        data_source = "transcript_partial"
        print(f"  ⚠ {transcript_warning}")

    # --- 口癖（分核心/扩展） ---
    core_counts: Dict[str, int] = defaultdict(int)
    ext_counts: Dict[str, int] = defaultdict(int)
    total_chars = 0
    for s in target:
        text = s["text"]
        total_chars += len(text)
        for fw in filler_core:
            core_counts[fw] += text.count(fw)
        for fw in filler_extended:
            ext_counts[fw] += text.count(fw)

    total_core = sum(core_counts.values())
    total_ext = sum(ext_counts.values())
    total_fillers = total_core + total_ext
    core_density = safe_float(total_core / (total_chars / 1000)) if total_chars > 0 else 0
    all_density = safe_float(total_fillers / (total_chars / 1000)) if total_chars > 0 else 0

    # 口癖组合模式
    combo_patterns: List[Dict[str, Any]] = []
    combos = ["就是说", "然后就是", "其实就", "那这个", "所以说", "对对对"]
    for s in target:
        for c in combos:
            if c in s["text"]:
                combo_patterns.append({"pattern": c, "time": fmt_time(s["start"])})

    # --- 语速 ---
    rates: List[float] = []
    for s in target:
        dur = s["end"] - s["start"]
        if dur > 0.5 and len(s["text"]) > 2:
            cpm = len(s["text"]) / dur * 60
            rates.append(cpm)

    rate_mean = safe_float(np.mean(rates)) if rates else 0
    rate_median = safe_float(np.median(rates)) if rates else 0
    rate_std = safe_float(np.std(rates)) if rates else 0
    rate_cv = safe_float(rate_std / rate_mean) if rate_mean > 0 else 0

    # --- 句长分布 ---
    lengths = [len(s["text"]) for s in target]
    sent_mean = safe_float(np.mean(lengths)) if lengths else 0
    sent_median = safe_float(np.median(lengths)) if lengths else 0
    over_100 = sum(1 for l in lengths if l > 100)
    under_10 = sum(1 for l in lengths if l < 10)
    frag_rate = safe_float(under_10 / len(lengths)) if lengths else 0

    # --- 停顿分布 ---
    gaps: List[float] = []
    for i in range(1, len(target)):
        gap = target[i]["start"] - target[i - 1]["end"]
        if gap > 0:
            gaps.append(gap)

    gap_median = safe_float(np.median(gaps)) if gaps else 0
    pause_short = sum(1 for g in gaps if g < 1.0)
    pause_medium = sum(1 for g in gaps if 1.0 <= g < 3.0)
    pause_long = sum(1 for g in gaps if 3.0 <= g < 5.0)
    pause_extra = sum(1 for g in gaps if g >= 5.0)

    # --- 发言总时长 ---
    speaking_time = sum(s["end"] - s["start"] for s in target)

    print(
        f"  核心口癖密度: {core_density}/千字, 全量密度: {all_density}/千字, "
        f"语速: {rate_mean}字/分, 句长: {sent_mean}字, 碎片率: {frag_rate:.1%}"
    )

    return {
        "meta": {
            "total_chars": total_chars,
            "speaking_seconds": safe_float(speaking_time),
            "segment_count": len(target),
            "data_source": data_source,
            "transcript_warning": transcript_warning,
        },
        "filler_words": {
            "core_total": total_core,
            "core_density_per_1000": core_density,
            "all_density_per_1000": all_density,
            "core_breakdown": dict(sorted(core_counts.items(), key=lambda x: -x[1])),
            "extended_breakdown": dict(sorted(ext_counts.items(), key=lambda x: -x[1])),
            "combos": combo_patterns[:10],
        },
        "speech_rate": {
            "mean_cpm": rate_mean,
            "median_cpm": rate_median,
            "std_cpm": rate_std,
            "cv": rate_cv,
        },
        "sentence_length": {
            "mean": sent_mean,
            "median": sent_median,
            "over_100": over_100,
            "under_10": under_10,
            "fragmentation_rate": frag_rate,
        },
        "pauses": {
            "median_gap_sec": gap_median,
            "short_lt1s": pause_short,
            "medium_1_3s": pause_medium,
            "long_3_5s": pause_long,
            "extra_long_gt5s": pause_extra,
        },
    }
