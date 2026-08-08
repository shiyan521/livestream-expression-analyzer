"""
声学特征提取模块。

包含以下功能:
- F0 (yin)、RMS 能量、频谱质心/平坦度/带宽、ZCR、Onset 强度提取
- 发音清晰度计算
- 分频带能量计算
- 非语音事件检测（静音/笑声/呼吸/爆音）
"""

import time
from typing import Any, Dict, List

import librosa
import numpy as np

from analyzer.utils import HOP, SR, fmt_time, safe_float


def compute_acoustic_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """一次性计算全部声学特征，返回特征字典。

    Args:
        y: 音频波形数组。
        sr: 采样率。

    Returns:
        包含所有声学特征的特征字典，键包括:
            f0, f0_valid, f0_times, rms, rms_times, rms_db,
            S, spectral_centroid, spectral_flatness, spectral_bandwidth,
            zcr, onset_env, mid_energy, low_energy, high_energy,
            total_spectral_energy, mid_low_ratio, hf_ratio, freqs,
            duration, sr
    """
    print("[3/9] 声学特征提取...")
    t0 = time.time()
    n = len(y)
    duration = n / sr
    print(f"  音频: {duration:.1f}s, sr={sr}")

    # --- F0 (yin, 比 pyin 快 3x) ---
    print("  F0 (yin)...")
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr, hop_length=HOP)
    f0_valid = f0[~np.isnan(f0)]
    f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)

    # --- RMS 能量 ---
    print("  RMS energy...")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    rms_db = librosa.amplitude_to_db(rms)

    # --- 频谱特征 ---
    print("  Spectral features...")
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]

    # --- Onset 强度 ---
    print("  Onset strength...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    # --- 频谱分频带能量（用于笑声/呼吸检测）---
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mid_mask = (freqs >= 1000) & (freqs <= 4000)
    low_mask = (freqs >= 100) & (freqs <= 500)
    high_mask = (freqs >= 4000) & (freqs <= 8000)
    mid_energy = np.sum(S[mid_mask, :] ** 2, axis=0)
    low_energy = np.sum(S[low_mask, :] ** 2, axis=0)
    high_energy = np.sum(S[high_mask, :] ** 2, axis=0)
    total_spectral_energy = np.sum(S ** 2, axis=0)

    # 中频/低频比（笑声音征）
    mid_low_ratio = np.zeros_like(mid_energy)
    valid_low = low_energy > 0
    mid_low_ratio[valid_low] = mid_energy[valid_low] / low_energy[valid_low]

    # 高频比（呼吸音征）
    hf_ratio = np.zeros_like(total_spectral_energy)
    valid_total = total_spectral_energy > 0
    hf_ratio[valid_total] = high_energy[valid_total] / total_spectral_energy[valid_total]

    elapsed = time.time() - t0
    print(f"  声学特征完成, 耗时 {elapsed:.0f}s")

    return {
        "f0": f0,
        "f0_valid": f0_valid,
        "f0_times": f0_times,
        "rms": rms,
        "rms_times": rms_times,
        "rms_db": rms_db,
        "S": S,
        "spectral_centroid": spectral_centroid,
        "spectral_flatness": spectral_flatness,
        "spectral_bandwidth": spectral_bandwidth,
        "zcr": zcr,
        "onset_env": onset_env,
        "mid_energy": mid_energy,
        "low_energy": low_energy,
        "high_energy": high_energy,
        "total_spectral_energy": total_spectral_energy,
        "mid_low_ratio": mid_low_ratio,
        "hf_ratio": hf_ratio,
        "freqs": freqs,
        "duration": duration,
        "sr": sr,
    }


def compute_clarity(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    feats: Dict[str, Any],
) -> Dict[str, Any]:
    """基于频谱平坦度 + 字均时长评估发音清晰度。

    Args:
        sentences: ASR 句子列表，每个句子包含 start, end, text, speaker 字段。
        target_speaker: 目标说话人 ID。
        feats: 声学特征字典（来自 compute_acoustic_features）。

    Returns:
        清晰度分析结果字典，包含:
            clarity_score: 整体清晰度均分 (0-100)
            total_segments: 分析的句子数
            slow_words: 语速过慢的句子列表
            swallowed_words: 吞字的句子列表
    """
    print("[4/9] 发音清晰度分析...")
    sf = feats["spectral_flatness"]
    f0_times = feats["f0_times"]

    target_sents = [s for s in sentences if s["speaker"] == target_speaker]
    clarity_scores: List[float] = []
    slow_words: List[Dict[str, Any]] = []
    fast_words: List[Dict[str, Any]] = []

    for s in target_sents:
        t0, t1 = s["start"], s["end"]
        dur = t1 - t0
        text = s["text"]
        n_chars = max(len(text), 1)
        char_dur = dur / n_chars  # 每字秒数

        # 该句时间范围内的频谱平坦度均值
        # 注意：spectral_flatness 的时间轴与 STFT frames 对应
        # 使用 f0_times 作为近似时间轴（两者 hop_length 相同）
        mask = (f0_times >= t0) & (f0_times < t1)
        sf_seg = sf[mask] if mask.sum() > 0 else np.array([0])
        mean_sf = float(np.mean(sf_seg))

        # 清晰度分: 100=完美, 按平坦度和字时长扣分
        score = 100
        if mean_sf > 0.05:
            score -= min(40, (mean_sf - 0.05) * 400)
        if char_dur > 0.34:  # >340ms/字 = 慢/不清
            score -= min(30, (char_dur - 0.34) * 100)
        if char_dur < 0.08:  # <80ms/字 = 吞字
            score -= min(20, (0.08 - char_dur) * 200)
        score = max(0, min(100, score))
        clarity_scores.append(score)

        if char_dur > 0.34:
            slow_words.append({
                "text": text[:30],
                "time": fmt_time(t0),
                "char_ms": int(char_dur * 1000),
                "flatness": safe_float(mean_sf, 4),
            })
        elif char_dur < 0.08:
            fast_words.append({
                "text": text[:30],
                "time": fmt_time(t0),
                "char_ms": int(char_dur * 1000),
            })

    overall = safe_float(np.mean(clarity_scores)) if clarity_scores else 0
    print(f"  清晰度均分: {overall}, 慢词: {len(slow_words)}, 快词: {len(fast_words)}")

    return {
        "clarity_score": overall,
        "total_segments": len(clarity_scores),
        "slow_words": slow_words[:10],
        "swallowed_words": fast_words[:10],
    }


def compute_non_speech_events(feats: Dict[str, Any]) -> Dict[str, Any]:
    """静音/笑声/呼吸/爆音检测。

    基于声学特征检测非语音事件:
    - 静音: RMS < -40dB 且持续 >3s
    - 笑声: 0.3-4s, 中频能量高, mid/low ratio 高
    - 呼吸: 短, 高频比高
    - 爆音: 极强 onset + 高频谱平坦度

    Args:
        feats: 声学特征字典（来自 compute_acoustic_features）。

    Returns:
        非语音事件检测结果字典。
    """
    print("[8/9] 非语音事件检测...")
    rms = feats["rms"]
    rms_db = feats["rms_db"]
    rms_times = feats["rms_times"]
    mid_low_ratio = feats["mid_low_ratio"]
    hf_ratio = feats["hf_ratio"]
    S = feats["S"]
    onset_env = feats["onset_env"]
    sr = feats["sr"]
    duration = feats["duration"]

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)

    # --- 静音 (>3s) ---
    silence_threshold = -40
    silent_frames = rms_db < silence_threshold
    silences: List[Dict[str, Any]] = []
    in_sil = False
    sil_start = 0
    for i, (t, is_s) in enumerate(zip(times, silent_frames)):
        if is_s and not in_sil:
            sil_start = t
            in_sil = True
        elif not is_s and in_sil:
            dur = t - sil_start
            if dur >= 3.0:
                silences.append({
                    "start": fmt_time(sil_start),
                    "duration_sec": safe_float(dur),
                })
            in_sil = False
    if in_sil:
        dur = duration - sil_start
        if dur >= 3.0:
            silences.append({
                "start": fmt_time(sil_start),
                "duration_sec": safe_float(dur),
            })

    # --- 笑声 (0.3-4s, 中频能量高, mid/low ratio高) ---
    mid_smooth = np.convolve(mid_low_ratio, np.ones(5) / 5, mode="same")
    nonzero = mid_smooth[mid_smooth > 0]
    ratio_thresh = np.percentile(nonzero, 85) if len(nonzero) > 0 else 1.0
    median_e = np.median(mid_smooth[mid_smooth > 0]) if np.any(mid_smooth > 0) else 0
    energy_thresh = median_e * 3  # 提高到3x，减少误检
    laugh_cand = (mid_smooth > energy_thresh) & (mid_low_ratio > ratio_thresh)
    laugh_times = librosa.frames_to_time(np.arange(len(laugh_cand)), sr=sr, hop_length=HOP)

    laughter: List[Dict[str, Any]] = []
    in_r = False
    r_start = 0
    for i, (t, is_c) in enumerate(zip(laugh_times, laugh_cand)):
        if is_c and not in_r:
            r_start = t
            in_r = True
        elif not is_c and in_r:
            dur = t - r_start
            if 0.3 <= dur <= 4.0:
                laughter.append({
                    "time": fmt_time(r_start),
                    "duration_sec": safe_float(dur),
                })
            in_r = False

    # --- 呼吸/叹气 (短, 高频比高) ---
    nonzero_total = feats["total_spectral_energy"][feats["total_spectral_energy"] > 0]
    low_e_thresh = np.percentile(nonzero_total, 10) if len(nonzero_total) > 0 else 0  # 更严格
    low_e_frames = feats["total_spectral_energy"] < low_e_thresh
    hf_thresh = np.percentile(hf_ratio[hf_ratio > 0], 90) if np.any(hf_ratio > 0) else 0.5  # 更严格
    breath_cand = low_e_frames & (hf_ratio > hf_thresh)

    breaths: List[Dict[str, Any]] = []
    in_b = False
    b_start = 0
    for i, (t, is_c) in enumerate(zip(times, breath_cand)):
        if is_c and not in_b:
            b_start = t
            in_b = True
        elif not is_c and in_b:
            dur = t - b_start
            if 0.05 <= dur <= 0.8:
                breaths.append({
                    "time": fmt_time(b_start),
                    "duration_sec": safe_float(dur),
                })
            in_b = False

    # --- 爆音 (极强 onset + 高频谱平坦度) ---
    onset_frames = librosa.onset.onset_detect(
        y=None,
        sr=sr,
        hop_length=HOP,
        onset_envelope=onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=10,
        post_avg=10,
        delta=0.5,
        wait=10,
    )
    onset_strength_vals = (
        onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
    )
    pops: List[Dict[str, Any]] = []
    if len(onset_strength_vals) > 0:
        pop_thresh = (
            np.percentile(onset_strength_vals[onset_strength_vals > 0], 99)
            if np.any(onset_strength_vals > 0)
            else 999
        )
        for idx, f_idx in enumerate(onset_frames):
            if onset_strength_vals[idx] > pop_thresh:
                t = librosa.frames_to_time(f_idx, sr=sr, hop_length=HOP)
                sf_start = max(0, f_idx - 1)
                sf_end = min(S.shape[1], f_idx + 2)
                if sf_end > sf_start:
                    sf_val = float(
                        np.mean(
                            librosa.feature.spectral_flatness(S=S[:, sf_start:sf_end])
                        )
                    )
                    if sf_val > 0.3:
                        pops.append({
                            "time": fmt_time(t),
                            "flatness": safe_float(sf_val, 4),
                        })

    print(
        f"  静音: {len(silences)}, 笑声: {len(laughter)}, "
        f"呼吸: {len(breaths)}, 爆音: {len(pops)}"
    )

    return {
        "silences_gt3s": silences[:30],
        "laughter": laughter[:30],
        "breaths": breaths[:30],
        "pops": pops[:20],
        "summary": {
            "silences": len(silences),
            "laughter": len(laughter),
            "breaths": len(breaths),
            "pops": len(pops),
            "note": "多人混音音频时，笑声/呼吸等非语音事件无法精确归属到具体说话人，数据为全音频混合检测结果",
        },
    }
