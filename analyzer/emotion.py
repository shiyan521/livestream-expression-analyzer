"""
情绪曲线与疲劳检测模块。

包含以下功能:
- 2 分钟窗口情绪评分（F0 + RMS + 语速加权）
- 前后半段对比分析
- 疲劳信号检测（语速/能量/音高下降）
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from analyzer.utils import fmt_time, safe_float


def compute_emotion_and_fatigue(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    feats: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """每 2 分钟窗口的情绪曲线 + 前后半段疲劳对比。

    情绪评分公式: 0.3*F0_norm + 0.4*RMS_norm + 0.3*rate_norm
    疲劳检测: 比较前半段与后半段的情绪/F0/能量/语速变化百分比。

    Args:
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        feats: 声学特征字典（来自 compute_acoustic_features）。

    Returns:
        元组 (emotion_result, emotion_curve):
            emotion_result: 情绪分析结果字典，包含 emotion_curve、
                emotion_peaks、emotion_valleys、fatigue 子字典
            emotion_curve: 情绪曲线列表（与 emotion_result 中的相同，
                单独返回便于管线其他步骤使用）
    """
    print("[6/9] 情绪曲线 + 疲劳检测...")
    f0 = feats["f0"]
    f0_times = feats["f0_times"]
    rms = feats["rms"]
    rms_times = feats["rms_times"]
    duration = feats["duration"]

    target = [s for s in sentences if s["speaker"] == target_speaker]
    if not target:
        return {"emotion_curve": [], "fatigue": {}}, []

    # 归一化 F0 和 RMS
    f0_valid = f0[~np.isnan(f0)]
    f0_min, f0_max = (
        (np.min(f0_valid), np.max(f0_valid)) if len(f0_valid) > 0 else (0, 1)
    )
    rms_nonzero = rms[rms > 0]
    rms_min, rms_max = (
        (np.min(rms_nonzero), np.max(rms_nonzero)) if len(rms_nonzero) > 0 else (0, 1)
    )

    window_sec = 120  # 2分钟窗口
    n_windows = max(1, int(duration / window_sec))
    emotion_curve: List[Dict[str, Any]] = []

    for w in range(n_windows):
        w_start = w * window_sec
        w_end = min((w + 1) * window_sec, duration)

        # F0 均值
        mask_f0 = (f0_times >= w_start) & (f0_times < w_end)
        f0_seg = f0[mask_f0]
        f0_seg = f0_seg[~np.isnan(f0_seg)]
        f0_norm = safe_float(
            (np.mean(f0_seg) - f0_min) / (f0_max - f0_min + 1e-9)
        )

        # RMS 均值
        mask_rms = (rms_times >= w_start) & (rms_times < w_end)
        rms_seg = rms[mask_rms]
        rms_norm = safe_float(
            (np.mean(rms_seg) - rms_min) / (rms_max - rms_min + 1e-9)
        )

        # 语速 (该窗口内目标说话人的字/秒)
        win_sents = [s for s in target if w_start <= s["start"] < w_end]
        chars = sum(len(s["text"]) for s in win_sents)
        speak_time = sum(s["end"] - s["start"] for s in win_sents)
        rate_cps = chars / speak_time if speak_time > 0 else 0
        rate_norm = min(1.0, rate_cps / 8.0)  # 8字/秒为满分

        score = 0.3 * f0_norm + 0.4 * rms_norm + 0.3 * rate_norm
        emotion_curve.append({
            "time_min": w * 2,
            "time_range": f"{fmt_time(w_start)}-{fmt_time(w_end)}",
            "score": safe_float(score * 100),
            "f0_norm": f0_norm,
            "energy_norm": rms_norm,
            "rate_norm": safe_float(rate_norm),
        })

    # 疲劳检测: 前半段 vs 后半段
    mid = len(emotion_curve) // 2
    first_half = emotion_curve[:mid]
    second_half = emotion_curve[mid:]

    def half_stats(half: List[Dict[str, Any]]) -> Dict[str, float]:
        if not half:
            return {}
        scores = [h["score"] for h in half]
        f0s = [h["f0_norm"] for h in half]
        energies = [h["energy_norm"] for h in half]
        rates = [h["rate_norm"] for h in half]
        return {
            "emotion_mean": safe_float(np.mean(scores)),
            "f0_mean": safe_float(np.mean(f0s)),
            "energy_mean": safe_float(np.mean(energies)),
            "rate_mean": safe_float(np.mean(rates)),
        }

    fh = half_stats(first_half)
    sh = half_stats(second_half)
    changes: Dict[str, float] = {}
    for key in ["emotion_mean", "f0_mean", "energy_mean", "rate_mean"]:
        v1 = fh.get(key, 0)
        v2 = sh.get(key, 0)
        if v1 > 0:
            changes[f"{key}_pct"] = safe_float((v2 - v1) / v1 * 100)

    # 峰值/低谷
    sorted_curve = sorted(emotion_curve, key=lambda x: x["score"], reverse=True)
    peaks = sorted_curve[:3]
    valleys = sorted_curve[-2:] if len(sorted_curve) >= 2 else []

    # 疲劳信号统计
    fatigue_flags: List[str] = []
    if changes.get("rate_mean_pct", 0) < -10:
        fatigue_flags.append("语速下降>10%")
    if changes.get("energy_mean_pct", 0) < -10:
        fatigue_flags.append("能量下降>10%")
    if changes.get("f0_mean_pct", 0) < -10:
        fatigue_flags.append("音高下降>10%")

    print(
        f"  情绪窗口: {len(emotion_curve)}, "
        f"峰值: {peaks[0]['score'] if peaks else 0}, "
        f"疲劳信号: {len(fatigue_flags)}"
    )

    return {
        "emotion_curve": emotion_curve,
        "emotion_peaks": [
            {"time": p["time_range"], "score": p["score"]} for p in peaks
        ],
        "emotion_valleys": [
            {"time": v["time_range"], "score": v["score"]} for v in valleys
        ],
        "fatigue": {
            "first_half": fh,
            "second_half": sh,
            "changes_pct": changes,
            "fatigue_flags": fatigue_flags,
        },
    }, emotion_curve
