#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发音诊断模块 - 直播音频深层发音分析。

七个分析维度：
  1. 尾音消失检测 (Tail Drop Detection)
  2. 咬字紧张度 (Articulation Tension)
  3. 语速-清晰度联动分析 (Speed-Clearness Correlation)
  4. 音高稳定性 / Jitter (Pitch Stability)
  5. 呼吸节奏分析 (Breathing Pattern)
  6. 长句尾段能量衰减 (Long Sentence Tail)
  7. 英文词汇发音检测 (English Word Detection)

依赖:
  - librosa (音频特征提取)
  - numpy (数值计算)

用法:
  python -m diagnosis.pronunciation --audio <audio.mp3> --asr-json <asr.json> [--speaker auto] --output result.json
"""
import argparse
import json
import re
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

# 目标说话人相关关键词（用于自动检测 speaker_id）
DEFAULT_SPEAKER_KEYWORDS: List[str] = [
    # 在此配置目标说话人的关键词，用于自动检测 speaker_id
    # 示例：["你的名字", "你的产品名", "你的公司名"]
]

MIN_SENTENCE_DURATION: float = 0.5   # 秒，低于此值的句子跳过
MIN_SENTENCE_CHARS: int = 3          # 字符数，低于此值的句子跳过
DEFAULT_SAMPLE_RATE: int = 22050     # 默认采样率


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def load_asr_data(asr_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """加载所有ASR JSON文件，合并sentences。

    Args:
        asr_paths: ASR JSON 文件路径列表

    Returns:
        合并后的句子列表，每个元素是原始 ASR sentence 字典
    """
    all_sentences: List[Dict[str, Any]] = []
    for path in asr_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for transcript in data.get("transcripts", []):
            for sent in transcript.get("sentences", []):
                all_sentences.append(sent)
    return all_sentences


def detect_speaker_id(
    sentences: List[Dict[str, Any]],
    keywords: Optional[List[str]] = None,
) -> int:
    """自动检测目标说话人的 speaker_id。

    策略：对每个 speaker_id 统计关键词命中次数，命中最多的即为目标说话人。

    Args:
        sentences: ASR 句子列表
        keywords: 说话人识别关键词列表，默认使用 DEFAULT_SPEAKER_KEYWORDS

    Returns:
        检测到的说话人 ID
    """
    if keywords is None:
        keywords = DEFAULT_SPEAKER_KEYWORDS

    speaker_keyword_count: Dict[int, int] = {}
    for sent in sentences:
        sid = sent.get("speaker_id", -1)
        text = sent.get("text", "")
        if sid not in speaker_keyword_count:
            speaker_keyword_count[sid] = 0
        for kw in keywords:
            if kw in text:
                speaker_keyword_count[sid] += 1
    if not speaker_keyword_count:
        print("[WARNING] 未检测到任何speaker，默认使用 speaker_id=0")
        return 0
    best_sid = max(speaker_keyword_count, key=speaker_keyword_count.get)
    print(f"[INFO] 说话人识别结果: speaker_id={best_sid} "
          f"(关键词命中 {speaker_keyword_count[best_sid]} 次)")
    return best_sid


def get_speaker_sentences(
    sentences: List[Dict[str, Any]],
    speaker_id: int,
    min_duration: float = MIN_SENTENCE_DURATION,
    min_chars: int = MIN_SENTENCE_CHARS,
) -> List[Dict[str, Any]]:
    """提取目标说话人的句子，并过滤掉过短的句子。

    Args:
        sentences: ASR 句子列表
        speaker_id: 目标说话人 ID
        min_duration: 最小句子时长（秒），低于此值跳过
        min_chars: 最小字符数，低于此值跳过

    Returns:
        过滤后的句子列表，每个元素包含 begin, end, duration, text,
        begin_ms, end_ms 字段
    """
    sents: List[Dict[str, Any]] = []
    for sent in sentences:
        if sent.get("speaker_id") != speaker_id:
            continue
        begin_s = sent["begin_time"] / 1000.0
        end_s = sent["end_time"] / 1000.0
        duration = end_s - begin_s
        text = sent.get("text", "").strip()
        # 过滤条件
        if duration < min_duration:
            continue
        if len(text) < min_chars:
            continue
        sents.append({
            "begin": begin_s,
            "end": end_s,
            "duration": duration,
            "text": text,
            "begin_ms": sent["begin_time"],
            "end_ms": sent["end_time"],
        })
    return sents


def format_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS。

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def extract_audio_segment(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
) -> Optional[np.ndarray]:
    """提取指定时间段的音频。

    Args:
        y: 音频信号数组
        sr: 采样率
        start_s: 开始时间（秒）
        end_s: 结束时间（秒）

    Returns:
        音频片段数组，如果无效则返回 None
    """
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    if end_sample > len(y):
        end_sample = len(y)
    if start_sample >= end_sample:
        return None
    return y[start_sample:end_sample]


def compute_rms(y_segment: Optional[np.ndarray]) -> float:
    """计算RMS能量。

    Args:
        y_segment: 音频片段数组

    Returns:
        RMS 能量值，无效输入返回 0.0
    """
    if y_segment is None or len(y_segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y_segment ** 2)))


def compute_rms_over_time(
    y: np.ndarray,
    sr: int,
    frame_len: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """按时间帧计算RMS能量序列。

    Args:
        y: 音频信号数组
        sr: 采样率
        frame_len: 帧长度（秒），默认 0.05

    Returns:
        (rms_values, time_points) 元组，均为 numpy 数组
    """
    hop = int(frame_len * sr)
    rms_vals: List[float] = []
    time_pts: List[float] = []
    for i in range(0, len(y), hop):
        frame = y[i:i + hop]
        if len(frame) == 0:
            break
        rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))
        time_pts.append(i / sr)
    return np.array(rms_vals), np.array(time_pts)


def compute_spectral_centroid_mean(y: Optional[np.ndarray], sr: int) -> float:
    """计算spectral centroid均值 (Hz)。

    Args:
        y: 音频信号数组
        sr: 采样率

    Returns:
        频谱质心均值，无效输入返回 0.0
    """
    if y is None or len(y) < 512:
        return 0.0
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.nanmean(cent))


def compute_spectral_flatness_mean(y: Optional[np.ndarray], sr: int) -> float:
    """计算spectral flatness均值。

    Args:
        y: 音频信号数组
        sr: 采样率

    Returns:
        频谱平坦度均值，无效输入返回 0.0
    """
    if y is None or len(y) < 512:
        return 0.0
    flat = librosa.feature.spectral_flatness(y=y)
    return float(np.nanmean(flat))


def compute_f0_cv(
    y: Optional[np.ndarray],
    sr: int,
) -> Tuple[float, float, float, float]:
    """计算F0的变异系数 (Coefficient of Variation)。

    Args:
        y: 音频信号数组
        sr: 采样率

    Returns:
        (cv, f0_mean, f0_std, voiced_ratio) 元组，
        cv 为变异系数，f0_mean 为基频均值，f0_std 为基频标准差，
        voiced_ratio 为有声帧比例
    """
    if y is None or len(y) < 512:
        return 0.0, 0.0, 0.0, 0.0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=80, fmax=500, sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)]
    voiced_ratio = len(voiced_f0) / max(len(f0), 1)
    if len(voiced_f0) < 3:
        return 0.0, 0.0, 0.0, voiced_ratio
    f0_mean = float(np.mean(voiced_f0))
    f0_std = float(np.std(voiced_f0))
    cv = f0_std / f0_mean if f0_mean > 0 else 0.0
    return cv, f0_mean, f0_std, voiced_ratio


def find_breaths(y: np.ndarray, sr: int) -> List[float]:
    """简单呼吸检测：在整段音频中检测短促低能量脉冲。

    策略：对比短时能量与周围能量的落差，落差大且持续很短的视为呼吸。

    Args:
        y: 音频信号数组
        sr: 采样率

    Returns:
        呼吸时间点列表（秒）
    """
    frame_len = 0.02  # 20ms帧
    hop = int(frame_len * sr)
    rms_vals: List[float] = []
    time_pts: List[float] = []
    for i in range(0, len(y), hop):
        frame = y[i:i + hop]
        rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))
        time_pts.append(i / sr)
    rms_arr = np.array(rms_vals)
    time_arr = np.array(time_pts)

    # 全局RMS的背景估计
    global_rms = float(np.median(rms_arr))
    # 呼吸特征：局部能量低于背景但周围有语音（能量先降后升）
    breath_points: List[float] = []
    window = 25  # 帧数 (~0.5秒)

    for i in range(window, len(rms_arr) - window):
        local_rms = rms_arr[i]
        # 呼吸：当前帧能量显著低于周围平均值
        surround_avg = (np.mean(rms_arr[i - window:i]) * 0.5
                        + np.mean(rms_arr[i + 1:i + window + 1]) * 0.5)
        if surround_avg > global_rms * 0.3 and local_rms < surround_avg * 0.3:
            # 确保是低谷（局部最小值附近）
            if (i == 0 or rms_arr[i] <= rms_arr[i - 1]) and \
               (i == len(rms_arr) - 1 or rms_arr[i] <= rms_arr[i + 1]):
                breath_points.append(float(time_arr[i]))

    # 合并相邻呼吸点（<0.3秒的合并）
    merged: List[float] = []
    for bp in breath_points:
        if not merged or bp - merged[-1] > 0.3:
            merged.append(bp)
    return merged


# ══════════════════════════════════════════════════════════════
# 分析模块
# ══════════════════════════════════════════════════════════════
def analysis_1_tail_drop(
    sents: List[Dict[str, Any]],
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """1. 尾音消失检测 (Tail Drop Detection)。

    通过比较每句末尾0.5秒和开头0.5秒的RMS能量比，
    检测句尾能量衰减（吞音/尾音消失）现象。

    Args:
        sents: 句子列表
        y: 音频信号数组
        sr: 采样率

    Returns:
        包含尾音消失统计和最严重例子的分析结果
    """
    print("\n" + "=" * 70)
    print("【1】尾音消失检测 (Tail Drop Detection)")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    for idx, s in enumerate(sents):
        seg = extract_audio_segment(y, sr, s["begin"], s["end"])
        if seg is None or len(seg) < sr * 0.5:
            continue

        tail_seg = seg[-int(0.5 * sr):]
        head_seg = seg[:int(0.5 * sr)]

        tail_rms = compute_rms(tail_seg)
        head_rms = compute_rms(head_seg)

        if head_rms < 1e-6:
            ratio = 1.0  # 避免除零
        else:
            ratio = tail_rms / head_rms

        results.append({
            "index": idx,
            "text": s["text"],
            "begin": s["begin"],
            "end": s["end"],
            "ratio": ratio,
            "tail_rms": tail_rms,
            "head_rms": head_rms,
        })

    # 分类
    normal = [r for r in results if r["ratio"] >= 0.3]
    mild = [r for r in results if 0.15 <= r["ratio"] < 0.3]
    severe = [r for r in results if r["ratio"] < 0.15]

    total = len(results)
    print(f"  总句数: {total}")
    print(f"  尾音消失 (ratio < 0.3): {len(mild)} 句 ({len(mild)/total*100:.1f}%)")
    print(f"  严重尾音消失 (ratio < 0.15): {len(severe)} 句 ({len(severe)/total*100:.1f}%)")
    print(f"  正常: {len(normal)} 句 ({len(normal)/total*100:.1f}%)")

    # 最严重的10个
    top_severe = sorted(results, key=lambda x: x["ratio"])[:10]
    print("\n  最严重的10个尾音消失例子:")
    print(f"  {'序号':>4}  {'时间':>8}  {'ratio':>8}  原文")
    print("  " + "-" * 80)
    for i, r in enumerate(top_severe):
        text_preview = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
        print(f"  {i+1:>4}  {format_time(r['begin']):>8}  {r['ratio']:>8.4f}  {text_preview}")

    return {
        "total": total,
        "tail_drop_count": len(mild),
        "tail_drop_ratio": len(mild) / total * 100 if total else 0,
        "severe_count": len(severe),
        "severe_ratio": len(severe) / total * 100 if total else 0,
        "normal_count": len(normal),
        "top10_severe": [
            {
                "time": format_time(r["begin"]),
                "ratio": round(r["ratio"], 4),
                "text": r["text"],
            }
            for r in top_severe
        ],
    }


def analysis_2_tension(
    sents: List[Dict[str, Any]],
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """2. 咬字紧张度 (Articulation Tension)。

    通过频谱质心 (spectral centroid) 评估发音紧张度，
    并分析紧张度随时间的变化趋势。

    Args:
        sents: 句子列表
        y: 音频信号数组
        sr: 采样率

    Returns:
        包含紧张度统计和时间趋势的分析结果
    """
    print("\n" + "=" * 70)
    print("【2】咬字紧张度 (Articulation Tension)")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    for idx, s in enumerate(sents):
        seg = extract_audio_segment(y, sr, s["begin"], s["end"])
        if seg is None or len(seg) < 512:
            continue
        centroid = compute_spectral_centroid_mean(seg, sr)
        results.append({
            "index": idx,
            "text": s["text"],
            "begin": s["begin"],
            "end": s["end"],
            "centroid": centroid,
        })

    tense = [r for r in results if r["centroid"] > 2500]
    relaxed = [r for r in results if r["centroid"] < 1500]
    normal = [r for r in results if 1500 <= r["centroid"] <= 2500]
    total = len(results)

    print(f"  总句数: {total}")
    print(f"  紧张 (>2500Hz): {len(tense)} 句 ({len(tense)/total*100:.1f}%)")
    print(f"  正常 (1500-2500Hz): {len(normal)} 句 ({len(normal)/total*100:.1f}%)")
    print(f"  松弛 (<1500Hz): {len(relaxed)} 句 ({len(relaxed)/total*100:.1f}%)")

    # 紧张度随时间的变化趋势（每10分钟窗口）
    windows: List[Dict[str, Any]] = []
    if results:
        max_time = max(r["end"] for r in results)
        window_size = 600  # 10分钟
        for w_start in np.arange(0, max_time, window_size):
            w_end = w_start + window_size
            w_sents = [r for r in results if r["begin"] >= w_start and r["begin"] < w_end]
            if w_sents:
                avg_centroid = np.mean([r["centroid"] for r in w_sents])
                windows.append({
                    "window_start_min": int(w_start / 60),
                    "window_end_min": int(w_end / 60),
                    "avg_centroid": round(float(avg_centroid), 1),
                    "sentence_count": len(w_sents),
                })
        print(f"\n  紧张度随时间变化趋势 (每10分钟窗口):")
        print(f"  {'时间段':>16}  {'平均centroid':>14}  {'句子数':>6}")
        print("  " + "-" * 42)
        for w in windows:
            label = f"{w['window_start_min']}-{w['window_end_min']}分钟"
            print(f"  {label:>16}  {w['avg_centroid']:>14.1f}  {w['sentence_count']:>6}")

    return {
        "total": total,
        "tense_count": len(tense),
        "tense_ratio": len(tense) / total * 100 if total else 0,
        "relaxed_count": len(relaxed),
        "relaxed_ratio": len(relaxed) / total * 100 if total else 0,
        "normal_count": len(normal),
        "time_trend": windows if results else [],
    }


def analysis_3_speed_clearness(
    sents: List[Dict[str, Any]],
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """3. 语速-清晰度联动分析 (Speed-Clearness Correlation)。

    按语速分组（慢/中/快），分析各组的频谱平坦度，
    判断语速快时是否存在吞音/不清晰现象。

    Args:
        sents: 句子列表
        y: 音频信号数组
        sr: 采样率

    Returns:
        包含各语速分组统计的分析结果
    """
    print("\n" + "=" * 70)
    print("【3】语速-清晰度联动分析 (Speed-Clearness Correlation)")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    for idx, s in enumerate(sents):
        seg = extract_audio_segment(y, sr, s["begin"], s["end"])
        if seg is None or len(seg) < 512:
            continue
        char_count = len(re.sub(r'\s+', '', s["text"]))
        duration_min = s["duration"] / 60.0
        speed = char_count / duration_min if duration_min > 0 else 0
        flatness = compute_spectral_flatness_mean(seg, sr)
        results.append({
            "index": idx,
            "text": s["text"],
            "begin": s["begin"],
            "end": s["end"],
            "speed": speed,       # 字/分钟
            "flatness": flatness,
            "char_count": char_count,
        })

    # 分组
    slow = [r for r in results if r["speed"] < 250]
    medium = [r for r in results if 250 <= r["speed"] <= 350]
    fast = [r for r in results if r["speed"] > 350]

    groups = [
        ("慢速 (<250字/分)", slow),
        ("中速 (250-350字/分)", medium),
        ("快速 (>350字/分)", fast),
    ]

    print(f"  总句数: {len(results)}\n")
    print(f"  {'语速分组':>20}  {'句数':>6}  {'平均flatness':>14}  {'平均语速':>12}")
    print("  " + "-" * 60)
    group_stats: List[Dict[str, Any]] = []
    for label, group in groups:
        if group:
            avg_flat = np.mean([r["flatness"] for r in group])
            avg_speed = np.mean([r["speed"] for r in group])
            print(f"  {label:>20}  {len(group):>6}  {avg_flat:>14.6f}  {avg_speed:>12.1f}")
            group_stats.append({
                "label": label,
                "count": len(group),
                "avg_flatness": round(float(avg_flat), 6),
                "avg_speed": round(float(avg_speed), 1),
            })
        else:
            print(f"  {label:>20}  {0:>6}  {'N/A':>14}  {'N/A':>12}")
            group_stats.append({
                "label": label,
                "count": 0,
                "avg_flatness": None,
                "avg_speed": None,
            })

    # 判断结论
    if len(slow) > 2 and len(fast) > 2:
        slow_flat = np.mean([r["flatness"] for r in slow])
        fast_flat = np.mean([r["flatness"] for r in fast])
        if fast_flat > slow_flat * 1.2:
            print(f"\n  [结论] 快速组flatness ({fast_flat:.6f}) 显著高于慢速组 ({slow_flat:.6f})，"
                  f"语速快时确实存在吞音/不清晰现象。")
        else:
            print(f"\n  [结论] 快速组flatness ({fast_flat:.6f}) 与慢速组 ({slow_flat:.6f}) 差异不大，"
                  f"语速对清晰度影响有限。")

    return {
        "total": len(results),
        "groups": group_stats,
    }


def analysis_4_pitch_stability(
    sents: List[Dict[str, Any]],
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """4. 音高稳定性 / Jitter (Pitch Stability)。

    使用 pyin 提取 F0，计算变异系数 (CV) 评估音高稳定性。
    CV < 0.05 为稳定，> 0.15 为抖动大。

    Args:
        sents: 句子列表
        y: 音频信号数组
        sr: 采样率

    Returns:
        包含音高稳定性统计和抖动最大例子的分析结果
    """
    print("\n" + "=" * 70)
    print("【4】音高稳定性 / Jitter (Pitch Stability)")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for idx, s in enumerate(sents):
        seg = extract_audio_segment(y, sr, s["begin"], s["end"])
        if seg is None or len(seg) < 512:
            continue
        # 只对长度>0.8秒的句子做F0分析（太短的不准确）
        if s["duration"] < 0.8:
            continue
        cv, f0_mean, f0_std, voiced_ratio = compute_f0_cv(seg, sr)
        results.append({
            "index": idx,
            "text": s["text"],
            "begin": s["begin"],
            "end": s["end"],
            "cv": cv,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "voiced_ratio": voiced_ratio,
        })
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  [进度] 已处理 {idx+1}/{len(sents)} 句, 耗时 {elapsed:.1f}s")

    # 分类
    stable = [r for r in results if r["cv"] < 0.05]
    normal_jitter = [r for r in results if 0.05 <= r["cv"] <= 0.15]
    unstable = [r for r in results if r["cv"] > 0.15]
    total = len(results)

    print(f"\n  总分析句数: {total}")
    print(f"  非常稳定 (CV < 0.05): {len(stable)} 句 ({len(stable)/total*100:.1f}%)")
    print(f"  正常波动 (CV 0.05-0.15): {len(normal_jitter)} 句 ({len(normal_jitter)/total*100:.1f}%)")
    print(f"  抖动大 (CV > 0.15): {len(unstable)} 句 ({len(unstable)/total*100:.1f}%)")

    # F0统计
    avg_f0: float = 0.0
    if results:
        valid_f0s = [r["f0_mean"] for r in results if r["f0_mean"] > 0]
        if valid_f0s:
            avg_f0 = float(np.mean(valid_f0s))
            print(f"  平均基频 F0: {avg_f0:.1f} Hz")

    # 抖动最严重的10个
    top_unstable = sorted(results, key=lambda x: x["cv"], reverse=True)[:10]
    if top_unstable:
        print(f"\n  抖动最大的10个句子:")
        print(f"  {'序号':>4}  {'时间':>8}  {'CV':>8}  {'F0均值':>8}  原文")
        print("  " + "-" * 80)
        for i, r in enumerate(top_unstable):
            text_preview = r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
            print(f"  {i+1:>4}  {format_time(r['begin']):>8}  {r['cv']:>8.4f}  "
                  f"{r['f0_mean']:>8.1f}  {text_preview}")

    return {
        "total": total,
        "stable_count": len(stable),
        "stable_ratio": len(stable) / total * 100 if total else 0,
        "normal_count": len(normal_jitter),
        "normal_ratio": len(normal_jitter) / total * 100 if total else 0,
        "unstable_count": len(unstable),
        "unstable_ratio": len(unstable) / total * 100 if total else 0,
        "avg_f0": round(avg_f0, 1) if results else 0,
        "top10_unstable": [
            {
                "time": format_time(r["begin"]),
                "cv": round(r["cv"], 4),
                "f0_mean": round(r["f0_mean"], 1),
                "text": r["text"],
            }
            for r in top_unstable
        ],
    }


def analysis_5_breathing(
    y: np.ndarray,
    sr: int,
    sents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """5. 呼吸节奏分析 (Breathing Pattern)。

    通过检测短促低能量脉冲来识别呼吸点，
    分析呼吸间隔模式（频繁/正常/稀少）。

    Args:
        y: 音频信号数组
        sr: 采样率
        sents: 句子列表（用于确定语音区间）

    Returns:
        包含呼吸节奏统计的分析结果
    """
    print("\n" + "=" * 70)
    print("【5】呼吸节奏分析 (Breathing Pattern)")
    print("=" * 70)

    print("  正在检测呼吸点...")
    breath_points = find_breaths(y, sr)
    print(f"  检测到 {len(breath_points)} 个呼吸点")

    if len(breath_points) < 2:
        print("  [WARNING] 呼吸点太少，无法进行间隔分析")
        return {
            "total_breaths": len(breath_points),
            "intervals": [],
        }

    # 计算相邻呼吸间隔
    intervals = np.diff(breath_points)
    # 只取语音区间内的呼吸（取目标说话人最早和最晚说话时间）
    speech_start = min(s["begin"] for s in sents)
    speech_end = max(s["end"] for s in sents)
    speech_intervals = [float(iv) for i, iv in enumerate(intervals)
                        if breath_points[i] >= speech_start and breath_points[i] <= speech_end]

    if not speech_intervals:
        speech_intervals = list(float(iv) for iv in intervals)

    frequent = [iv for iv in speech_intervals if iv < 5]
    normal_br = [iv for iv in speech_intervals if 5 <= iv <= 30]
    sparse = [iv for iv in speech_intervals if iv > 30]
    total_iv = len(speech_intervals)

    print(f"\n  语音区间内呼吸间隔分析:")
    print(f"  总间隔数: {total_iv}")
    print(f"  频繁换气 (<5秒): {len(frequent)} 次 ({len(frequent)/total_iv*100:.1f}%)")
    print(f"  正常 (5-30秒): {len(normal_br)} 次 ({len(normal_br)/total_iv*100:.1f}%)")
    print(f"  稀少换气 (>30秒): {len(sparse)} 次 ({len(sparse)/total_iv*100:.1f}%)")
    avg_iv: float = 0.0
    if speech_intervals:
        avg_iv = float(np.mean(speech_intervals))
        print(f"  平均呼吸间隔: {avg_iv:.1f} 秒")

    return {
        "total_breaths": len(breath_points),
        "total_intervals": total_iv,
        "frequent_count": len(frequent),
        "frequent_ratio": len(frequent) / total_iv * 100 if total_iv else 0,
        "normal_count": len(normal_br),
        "normal_ratio": len(normal_br) / total_iv * 100 if total_iv else 0,
        "sparse_count": len(sparse),
        "sparse_ratio": len(sparse) / total_iv * 100 if total_iv else 0,
        "avg_interval": round(avg_iv, 1) if speech_intervals else 0,
    }


def analysis_6_long_sentence_tail(
    sents: List[Dict[str, Any]],
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """6. 长句尾段能量衰减 (Long Sentence Tail)。

    筛选长度>50字的句子，分析其末尾20%时间段的RMS能量衰减斜率，
    检测长句尾段能量急剧衰减现象。

    Args:
        sents: 句子列表
        y: 音频信号数组
        sr: 采样率

    Returns:
        包含能量衰减统计和最严重例子的分析结果
    """
    print("\n" + "=" * 70)
    print("【6】长句尾段能量衰减 (Long Sentence Tail)")
    print("=" * 70)

    # 筛选长度>50字的句子
    long_sents = [s for s in sents if len(s["text"]) > 50]
    print(f"  总句数: {len(sents)}, 长句 (>50字): {len(long_sents)} 句")

    results: List[Dict[str, Any]] = []
    for idx, s in enumerate(long_sents):
        seg = extract_audio_segment(y, sr, s["begin"], s["end"])
        if seg is None or len(seg) < sr * 1.0:
            continue

        # 取最后20%时间段
        tail_start_sample = int(len(seg) * 0.8)
        tail_seg = seg[tail_start_sample:]

        # 计算RMS随时间的衰减斜率
        rms_vals, time_pts = compute_rms_over_time(tail_seg, sr, frame_len=0.05)
        if len(rms_vals) < 5:
            continue

        # 对RMS做线性回归求斜率
        x = np.arange(len(rms_vals))
        # 过滤零值避免log问题
        rms_safe = np.maximum(rms_vals, 1e-10)
        if np.max(rms_safe) < 1e-8:
            slope = 0.0
        else:
            slope = float(np.polyfit(x, rms_safe, 1)[0])  # 负值=衰减

        # 同时计算最后20%与前面80%的RMS比
        head_rms = compute_rms(seg[:tail_start_sample])
        tail_rms = compute_rms(tail_seg)
        ratio = tail_rms / head_rms if head_rms > 1e-6 else 1.0

        results.append({
            "index": idx,
            "text": s["text"],
            "begin": s["begin"],
            "end": s["end"],
            "char_count": len(s["text"]),
            "slope": slope,
            "head_rms": head_rms,
            "tail_rms": tail_rms,
            "ratio": ratio,
        })

    # 按slope排序（slope越小=衰减越快）
    degraded: List[Dict[str, Any]] = []
    top10: List[Dict[str, Any]] = []
    if results:
        # 阈值：slope的中位数 * 0.5 作为"急剧衰减"
        slopes = [r["slope"] for r in results]
        median_slope = float(np.median(slopes))
        threshold = median_slope * 0.5  # 更负

        degraded = [r for r in results if r["slope"] < threshold]
        print(f"  衰减斜率中位数: {median_slope:.6f}")
        print(f"  急剧衰减阈值: {threshold:.6f}")
        print(f"  急剧衰减长句数: {len(degraded)} 句 ({len(degraded)/len(results)*100:.1f}%)")

        top10 = sorted(results, key=lambda x: x["slope"])[:10]
        print(f"\n  能量衰减最严重的10个长句:")
        print(f"  {'序号':>4}  {'时间':>8}  {'斜率':>12}  {'尾/头比':>8}  {'字数':>4}  原文(前50字)")
        print("  " + "-" * 100)
        for i, r in enumerate(top10):
            text_preview = r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
            print(f"  {i+1:>4}  {format_time(r['begin']):>8}  {r['slope']:>12.6f}  "
                  f"{r['ratio']:>8.4f}  {r['char_count']:>4}  {text_preview}")

    return {
        "total_long_sentences": len(long_sents),
        "analyzed": len(results),
        "degraded_count": len(degraded),
        "degraded_ratio": len(degraded) / len(results) * 100 if results else 0,
        "top10": [
            {
                "time": format_time(r["begin"]),
                "slope": round(r["slope"], 6),
                "ratio": round(r["ratio"], 4),
                "char_count": r["char_count"],
                "text": r["text"],
            }
            for r in top10
        ],
    }


def analysis_7_english_words(
    sents: List[Dict[str, Any]],
    y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """7. 英文词汇发音检测 (English Word Detection)。

    从逐字稿中检测英文词汇，对比含英文句子与纯中文句子的声学特征
    （频谱质心、RMS能量），判断英文发音是否自信或含糊。

    Args:
        sents: 句子列表
        y: 音频信号数组
        sr: 采样率

    Returns:
        包含英文词频率统计和声学特征对比的分析结果
    """
    print("\n" + "=" * 70)
    print("【7】英文词汇发音检测 (English Word Detection)")
    print("=" * 70)

    # 正则匹配英文词
    eng_pattern = re.compile(r'[a-zA-Z]{2,}')
    eng_words_found: List[Dict[str, Any]] = []

    for idx, s in enumerate(sents):
        words = eng_pattern.findall(s["text"])
        for w in words:
            eng_words_found.append({
                "sentence_idx": idx,
                "word": w,
                "text": s["text"],
                "begin": s["begin"],
                "end": s["end"],
            })

    print(f"  在逐字稿中发现 {len(eng_words_found)} 处英文词汇")

    # 统计频率
    word_freq = Counter(w["word"].lower() for w in eng_words_found)
    print(f"  不重复英文词数: {len(word_freq)}")
    print(f"\n  出现频率最高的英文词:")
    for word, count in word_freq.most_common(20):
        print(f"    {word:>20}: {count} 次")

    # 对含英文的句子做acoustic对比分析
    eng_sents: List[Dict[str, Any]] = []
    for w_info in eng_words_found:
        seg = extract_audio_segment(y, sr, w_info["begin"], w_info["end"])
        if seg is None or len(seg) < 512:
            continue
        centroid = compute_spectral_centroid_mean(seg, sr)
        rms = compute_rms(seg)
        eng_sents.append({
            "word": w_info["word"],
            "text": w_info["text"],
            "time": format_time(w_info["begin"]),
            "centroid": round(centroid, 1),
            "rms": round(rms, 6),
        })

    # 对比纯中文句子
    cn_sents = [s for s in sents if not eng_pattern.search(s["text"])]
    cn_centroids: List[float] = []
    cn_rms_vals: List[float] = []
    for s in cn_sents:
        seg = extract_audio_segment(y, sr, s["begin"], s["end"])
        if seg is not None and len(seg) >= 512:
            cn_centroids.append(compute_spectral_centroid_mean(seg, sr))
            cn_rms_vals.append(compute_rms(seg))

    avg_cn_centroid: Optional[float] = None
    avg_cn_rms: Optional[float] = None
    avg_eng_centroid: Optional[float] = None
    avg_eng_rms: Optional[float] = None

    if cn_centroids and eng_sents:
        avg_cn_centroid = float(np.mean(cn_centroids))
        avg_cn_rms = float(np.mean(cn_rms_vals))
        avg_eng_centroid = float(np.mean([e["centroid"] for e in eng_sents]))
        avg_eng_rms = float(np.mean([e["rms"] for e in eng_sents]))

        print(f"\n  声学特征对比:")
        print(f"  {'':>20}  {'中文句子均值':>14}  {'含英文句子均值':>16}  {'差异':>10}")
        print("  " + "-" * 66)
        centroid_diff = avg_eng_centroid - avg_cn_centroid
        rms_diff_pct = (avg_eng_rms - avg_cn_rms) / avg_cn_rms * 100 if avg_cn_rms > 0 else 0
        print(f"  {'Spectral Centroid':>20}  {avg_cn_centroid:>14.1f}  {avg_eng_centroid:>16.1f}  {centroid_diff:>+10.1f}")
        print(f"  {'RMS Energy':>20}  {avg_cn_rms:>14.6f}  {avg_eng_rms:>16.6f}  {rms_diff_pct:>+9.1f}%")

        if avg_eng_rms < avg_cn_rms * 0.8:
            print(f"\n  [结论] 含英文段落的能量显著偏低 ({rms_diff_pct:+.1f}%)，"
                  f"可能存在英文发音不够自信或含糊的情况。")
        if abs(centroid_diff) > 200:
            direction = "偏高" if centroid_diff > 0 else "偏低"
            print(f"  [结论] 含英文段落的频谱质心{direction} ({centroid_diff:+.1f} Hz)，"
                  f"英文发音的频谱特征与中文有差异。")

    return {
        "total_english_words": len(eng_words_found),
        "unique_words": len(word_freq),
        "top_words": [
            {"word": w, "count": c} for w, c in word_freq.most_common(20)
        ],
        "eng_sentence_avg_centroid": round(avg_eng_centroid, 1) if cn_centroids and eng_sents else None,
        "cn_sentence_avg_centroid": round(avg_cn_centroid, 1) if cn_centroids else None,
        "eng_sentence_avg_rms": round(float(avg_eng_rms), 6) if cn_centroids and eng_sents else None,
        "cn_sentence_avg_rms": round(float(avg_cn_rms), 6) if cn_centroids else None,
    }


# ══════════════════════════════════════════════════════════════
# 完整诊断入口
# ══════════════════════════════════════════════════════════════
def run_pronunciation_diagnosis(
    audio_path: str,
    asr_paths: Sequence[str],
    speaker_id: Optional[int] = None,
    keywords: Optional[List[str]] = None,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, Any]:
    """执行完整的发音诊断流程。

    依次执行七个维度的发音分析，返回完整结果。

    Args:
        audio_path: 音频文件路径
        asr_paths: ASR JSON 文件路径列表
        speaker_id: 目标说话人ID，None 表示自动检测
        keywords: 说话人识别关键词列表
        sr: 采样率，默认 22050

    Returns:
        包含七个维度分析结果和元数据的完整诊断字典
    """
    total_start = time.time()

    print("=" * 70)
    print("  发音诊断分析")
    print("=" * 70)

    # ── 1. 加载ASR数据 ──
    print("\n[步骤1] 加载ASR数据...")
    all_sentences = load_asr_data(asr_paths)
    print(f"  总句子数: {len(all_sentences)}")

    # ── 2. 检测目标说话人的speaker_id ──
    print("\n[步骤2] 检测目标说话人的speaker_id...")
    if speaker_id is None:
        target_sid = detect_speaker_id(all_sentences, keywords)
    else:
        target_sid = speaker_id
        print(f"  使用指定 speaker_id={target_sid}")

    # ── 3. 提取目标说话人的句子 ──
    print("\n[步骤3] 提取目标说话人的句子...")
    target_sents = get_speaker_sentences(all_sentences, target_sid)
    print(f"  有效句子数: {len(target_sents)}")
    if target_sents:
        print(f"  时间跨度: {format_time(target_sents[0]['begin'])} ~ "
              f"{format_time(target_sents[-1]['end'])}")

    # ── 4. 加载音频 ──
    print(f"\n[步骤4] 加载音频文件...")
    print(f"  路径: {audio_path}")
    print("  正在加载 (librosa, sr=22050)，请耐心等待...")
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y) / sr_loaded
    print(f"  加载完成! 时长: {duration:.1f}秒 ({duration/60:.1f}分钟), "
          f"采样率: {sr_loaded}Hz")

    # ── 5. 逐项分析 ──
    output: Dict[str, Any] = {}

    output["tail_drop"] = analysis_1_tail_drop(target_sents, y, sr_loaded)
    output["tension"] = analysis_2_tension(target_sents, y, sr_loaded)
    output["speed_clearness"] = analysis_3_speed_clearness(target_sents, y, sr_loaded)
    output["pitch_stability"] = analysis_4_pitch_stability(target_sents, y, sr_loaded)
    output["breathing"] = analysis_5_breathing(y, sr_loaded, target_sents)
    output["long_sentence_tail"] = analysis_6_long_sentence_tail(target_sents, y, sr_loaded)
    output["english_words"] = analysis_7_english_words(target_sents, y, sr_loaded)

    # ── 6. 元数据 ──
    total_elapsed = time.time() - total_start
    output["metadata"] = {
        "audio_path": audio_path,
        "audio_duration_seconds": round(duration, 1),
        "sample_rate": sr_loaded,
        "target_speaker_id": target_sid,
        "target_sentence_count": len(target_sents),
        "analysis_time_seconds": round(total_elapsed, 1),
        "analysis_time_formatted": f"{int(total_elapsed//60)}分{int(total_elapsed%60)}秒",
    }

    return output


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════
def main() -> None:
    """命令行入口：解析参数并执行发音诊断。"""
    parser = argparse.ArgumentParser(description='发音诊断分析')
    parser.add_argument('--audio', required=True, help='音频文件路径')
    parser.add_argument('--asr-json', required=True, nargs='+', help='ASR JSON 文件路径（可多个）')
    parser.add_argument('--speaker', type=int, default=None, help='目标说话人 ID（不指定则自动检测）')
    parser.add_argument('--output', default='pronunciation_diagnosis.json', help='输出 JSON 路径')
    parser.add_argument('--sr', type=int, default=DEFAULT_SAMPLE_RATE, help='采样率')
    args = parser.parse_args()

    output = run_pronunciation_diagnosis(
        audio_path=args.audio,
        asr_paths=args.asr_json,
        speaker_id=args.speaker,
        sr=args.sr,
    )

    # 保存结果
    print("\n" + "=" * 70)
    print("  保存诊断结果...")
    print("=" * 70)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存到: {args.output}")

    # 综合总结
    print("\n" + "=" * 70)
    print("  综合诊断总结")
    print("=" * 70)

    r1 = output["tail_drop"]
    r2 = output["tension"]
    r3 = output["speed_clearness"]
    r4 = output["pitch_stability"]
    r5 = output["breathing"]
    r6 = output["long_sentence_tail"]
    r7 = output["english_words"]

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  1. 尾音消失: {r1['tail_drop_ratio']:.1f}% 句子尾音消失,
  │     {r1['severe_ratio']:.1f}% 为严重级别                          │
  │  2. 咬字紧张度: {r2['tense_ratio']:.1f}% 紧张, {r2['relaxed_ratio']:.1f}% 松弛
  │     建议关注紧张度时间趋势，看后半段是否有所改善          │
  │  3. 语速与清晰度: 快语速组flatness{('高于' if r3.get('groups') and len(r3['groups'])>=3 and r3['groups'][2].get('avg_flatness') and r3['groups'][0].get('avg_flatness') and r3['groups'][2]['avg_flatness'] > r3['groups'][0]['avg_flatness'] * 1.2 else '未显著高于')}慢语速组
  │  4. 音高稳定性: {r4['unstable_ratio']:.1f}% 句子抖动较大                     │
  │  5. 呼吸节奏: 频繁换气 {r5.get('frequent_ratio', 0):.1f}%, 稀少 {r5.get('sparse_ratio', 0):.1f}%       │
  │  6. 长句尾段: {r6.get('degraded_ratio', 0):.1f}% 长句出现能量急剧衰减                 │
  │  7. 英文词汇: 发现 {r7['total_english_words']} 处英文词,
  │     声学特征{('有' if r7.get('eng_sentence_avg_rms') and r7.get('cn_sentence_avg_rms') and r7['eng_sentence_avg_rms'] < r7['cn_sentence_avg_rms'] * 0.8 else '无显著')}异常                                │
  └─────────────────────────────────────────────────────────────┘
    """)

    total_elapsed = output["metadata"]["analysis_time_seconds"]
    print(f"\n  全部分析完成! 总耗时: {int(total_elapsed//60)}分{int(total_elapsed%60)}秒")


if __name__ == "__main__":
    main()
