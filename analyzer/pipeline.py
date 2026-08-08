"""
主分析管线模块。

从原 analyze.py 重构而来，包含:
- ASR + 说话人分离（通过 ASRBackend 接口调用，不再直接调用 bl CLI）
- ASR JSON 加载
- 目标说话人自动识别（关键词从 config/speakers.yaml 读取）
- 目标说话人音频段提取（ffmpeg）
- 主分析管线编排（声学/文本/情绪/互动/话题）
- 综合评分计算
- argparse 命令行入口

用法:
  python -m analyzer.pipeline --audio /path/to/audio.mp3 [--speakers 2] \\
      [--target-speaker auto] [--output result.json] [--clips-dir ./clips] \\
      [--top-clips 3] [--session-id 0601] [--skip-asr] [--asr-json existing.json]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np

from asr.base import ASRBackend

from analyzer.acoustic import (
    compute_acoustic_features,
    compute_clarity,
    compute_non_speech_events,
)
from analyzer.emotion import compute_emotion_and_fatigue
from analyzer.interaction import compute_interaction
from analyzer.text_metrics import compute_text_metrics
from analyzer.topics import compute_topics_and_clips
from analyzer.utils import (
    SR,
    fmt_time,
    load_filler_config,
    load_speaker_config,
    run_cmd,
    safe_float,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ASR + 说话人分离
# ═══════════════════════════════════════════════════════════════════════════════
def run_asr(
    audio_path: str,
    speaker_count: int,
    output_json: str,
    asr_backend: ASRBackend,
) -> str:
    """调用 ASR 后端做 ASR + 说话人分离，返回 JSON 路径。

    如果音频 > 45 分钟，自动分片处理避免超时。

    Args:
        audio_path: 音频文件路径。
        speaker_count: 预期说话人数量。
        output_json: 输出 JSON 文件路径。
        asr_backend: ASR 后端实例（ASRBackend 子类）。

    Returns:
        输出 JSON 文件路径。
    """
    print(f"[1/9] ASR + 说话人分离 (speakers={speaker_count})...")
    t0 = time.time()

    # 检测音频时长
    audio_duration = asr_backend.get_audio_duration(audio_path)
    print(f"  音频时长: {audio_duration:.0f}s ({audio_duration/60:.1f}min)")

    CHUNK_MIN = 45 * 60  # 45分钟

    if audio_duration <= CHUNK_MIN:
        # 正常单次 ASR
        asr_backend.recognize(audio_path, speaker_count, output_json)
    else:
        # 分片 ASR
        n_chunks = int(audio_duration // CHUNK_MIN) + 1
        chunk_dur = audio_duration / n_chunks
        print(f"  音频超过45分钟，自动分为 {n_chunks} 片 (每片 ~{chunk_dur/60:.0f}min)")

        chunk_dir = os.path.join(os.path.dirname(output_json), "asr_chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        all_sentences: List[Dict[str, Any]] = []
        for i in range(n_chunks):
            chunk_start = i * chunk_dur
            chunk_end = min((i + 1) * chunk_dur, audio_duration)
            chunk_file = os.path.join(chunk_dir, f"chunk_{i:02d}.mp3")
            chunk_asr = os.path.join(chunk_dir, f"chunk_{i:02d}_asr.json")

            # ffmpeg 切片
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", f"{chunk_start:.2f}", "-to", f"{chunk_end:.2f}",
                "-acodec", "copy", chunk_file,
            ]
            run_cmd(cmd, check=False)
            if not os.path.exists(chunk_file):
                print(f"  警告: 切片 {i} 失败，跳过", file=sys.stderr)
                continue

            # ASR（通过后端接口）
            asr_backend.recognize(chunk_file, speaker_count, chunk_asr)

            # 加载并偏移时间戳
            chunk_data = load_asr_json(chunk_asr)
            chunk_sents, _ = chunk_data
            for s in chunk_sents:
                s["start"] += chunk_start
                s["end"] += chunk_start
            all_sentences.extend(chunk_sents)
            print(f"  切片 {i+1}/{n_chunks}: {len(chunk_sents)} 句")

        # 合并并保存
        all_sentences.sort(key=lambda x: x["start"])
        merged_data = {
            "transcripts": [{"sentences": [
                {
                    "begin_time": int(s["start"] * 1000),
                    "end_time": int(s["end"] * 1000),
                    "text": s["text"],
                    "speaker_id": s["speaker"],
                    "words": s.get("words", []),
                } for s in all_sentences
            ]}],
            "properties": {
                "original_duration_in_milliseconds": int(audio_duration * 1000),
            },
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"  分片ASR合并完成: {len(all_sentences)} 句 -> {output_json}")

    elapsed = time.time() - t0
    print(f"  ASR 完成, 耗时 {elapsed:.0f}s -> {output_json}")
    return output_json


def load_asr_json(json_path: str) -> Tuple[List[Dict[str, Any]], float]:
    """加载 ASR JSON 文件，解析为句子列表。

    Args:
        json_path: ASR JSON 文件路径。

    Returns:
        元组 (sentences, duration):
            sentences: 句子列表，每个句子包含 start, end, text, speaker, words 字段
            duration: 音频总时长（秒）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences: List[Dict[str, Any]] = []
    for t in data.get("transcripts", []):
        for s in t.get("sentences", []):
            sentences.append({
                "start": s["begin_time"] / 1000.0,
                "end": s["end_time"] / 1000.0,
                "text": s["text"],
                "speaker": s["speaker_id"],
                "words": s.get("words", []),
            })
    sentences.sort(key=lambda x: x["start"])
    duration_ms = data.get("properties", {}).get(
        "original_duration_in_milliseconds", 0
    )
    return sentences, duration_ms / 1000.0 if duration_ms else 0


# ═══════════════════════════════════════════════════════════════════════════════
# 自动识别目标说话人
# ═══════════════════════════════════════════════════════════════════════════════
def auto_detect_speaker(
    sentences: List[Dict[str, Any]],
    speaker_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[int, Dict[str, Any]]]:
    """自动判断哪个 speaker_id 是目标说话人。

    三重验证策略:
    (1) 唯一关键词匹配（高区分度词，权重 x20）
    (2) 通用关键词辅助（权重 x5）
    (3) 反向关键词减分（命中减分 x15）
    (4) 置信度评估：分数差距小则警告
    (5) 打印双方样本供人工交叉确认

    关键词从 config/speakers.yaml 读取，不再硬编码个人关键词。

    Args:
        sentences: ASR 句子列表。
        speaker_config: 说话人配置字典，包含 target_speaker 子字典
            （unique_keywords, general_keywords, reverse_keywords）。
            若为 None 则从 config/speakers.yaml 加载。

    Returns:
        元组 (best_speaker_id, speaker_stats):
            best_speaker_id: 识别出的目标说话人 ID
            speaker_stats: 各说话人的统计信息字典
    """
    # 从配置加载关键词
    if speaker_config is None:
        speaker_config = load_speaker_config()

    target_cfg = speaker_config.get("target_speaker", {})
    speaker_name = target_cfg.get("name", "Speaker A")
    unique_keywords: List[str] = target_cfg.get("unique_keywords", [])
    common_keywords: List[str] = target_cfg.get("general_keywords", [])
    anti_keywords: List[str] = target_cfg.get("reverse_keywords", [])

    speaker_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "chars": 0, "time": 0, "segments": 0,
        "unique_hits": 0, "common_hits": 0, "anti_hits": 0, "questions": 0
    })
    for s in sentences:
        sp = s["speaker"]
        stats = speaker_stats[sp]
        stats["chars"] += len(s["text"])
        stats["time"] += s["end"] - s["start"]
        stats["segments"] += 1
        text_lower = s["text"].lower()
        # 唯一关键词（每个句子最多计1次）
        for kw in unique_keywords:
            if kw.lower() in text_lower:
                stats["unique_hits"] += 1
                break
        # 通用关键词（每个句子最多计1次）
        for kw in common_keywords:
            if kw.lower() in text_lower:
                stats["common_hits"] += 1
                break
        # 反向关键词（每个句子最多计1次）
        for kw in anti_keywords:
            if kw.lower() in text_lower:
                stats["anti_hits"] += 1
                break
        if "？" in s["text"] or "?" in s["text"]:
            stats["questions"] += 1

    if not speaker_stats:
        return 0, {}

    # 综合打分：唯一关键词x20 + 通用关键词x5 - 反向关键词x15 + 时间x0.005 + 问题x2
    scores: Dict[int, float] = {}
    best_sp = 0
    best_score = -999
    for sp, stats in speaker_stats.items():
        score = (
            stats["unique_hits"] * 20
            + stats["common_hits"] * 5
            - stats["anti_hits"] * 15
            + stats["time"] * 0.005
            + stats["questions"] * 2
        )
        scores[sp] = score
        if score > best_score:
            best_score = score
            best_sp = sp

    # 置信度评估：分数差距与最高分绝对值的比例
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    if len(sorted_scores) >= 2:
        score_gap = sorted_scores[0][1] - sorted_scores[1][1]
        max_abs = max(abs(s) for s in scores.values()) if scores else 1
        confidence = min(100, max(0, score_gap / max(max_abs, 1) * 100))
    else:
        confidence = 100

    print(
        f"  自动识别: speaker_id={best_sp} "
        f"(score={best_score:.1f}, confidence={confidence:.0f}%)"
    )
    for sp, stats in sorted(speaker_stats.items()):
        print(
            f"    speaker {sp}: time={stats['time']:.0f}s, "
            f"chars={stats['chars']}, "
            f"unique=+{stats['unique_hits']}, common=+{stats['common_hits']}, "
            f"anti=-{stats['anti_hits']}, questions={stats['questions']}, "
            f"score={scores[sp]:.1f}"
        )

    if confidence < 50:
        print(
            f"  ⚠️  置信度仅{confidence:.0f}%，speaker识别可能不准确！"
            f"请人工确认后使用 --target-speaker 指定正确ID"
        )

    # 打印前5句供人工确认
    target_sents = [s for s in sentences if s["speaker"] == best_sp][:5]
    if target_sents:
        print(f"  === speaker {best_sp} 前5句样本（请确认是否为{speaker_name}）===")
        for s in target_sents:
            print(f"    [{fmt_time(s['start'])}] {s['text'][:80]}")

    # 也打印另一说话人的样本供交叉验证
    other_sps = [sp for sp in speaker_stats if sp != best_sp]
    if other_sps:
        other_sents = [s for s in sentences if s["speaker"] == other_sps[0]][:3]
        if other_sents:
            print(f"  === speaker {other_sps[0]} 前3句样本（对比确认）===")
            for s in other_sents:
                print(f"    [{fmt_time(s['start'])}] {s['text'][:80]}")

    return best_sp, dict(speaker_stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 提取目标说话人音频段（ffmpeg）
# ═══════════════════════════════════════════════════════════════════════════════
def extract_speaker_audio(
    audio_path: str,
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    output_wav: str,
) -> float:
    """用 ffmpeg 把目标说话人的片段拼成一段连续音频。

    Args:
        audio_path: 原始音频文件路径。
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        output_wav: 输出 WAV 文件路径。

    Returns:
        提取的总时长（秒），未找到片段时返回 0。
    """
    print(f"[2/9] 提取 speaker {target_speaker} 的音频段...")
    t0 = time.time()

    # 生成 ffmpeg filter_complex 的时间段列表
    segments = [
        (s["start"], s["end"])
        for s in sentences
        if s["speaker"] == target_speaker
    ]
    if not segments:
        print("  警告: 未找到目标说话人片段", file=sys.stderr)
        return 0

    # 合并重叠/相邻段（间隔 < 0.5s）
    merged = [segments[0]]
    for start, end in segments[1:]:
        if start - merged[-1][1] < 0.5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # 用 ffmpeg concat
    with tempfile.TemporaryDirectory() as tmpdir:
        parts: List[str] = []
        for i, (start, end) in enumerate(merged):
            part_path = os.path.join(tmpdir, f"part_{i:04d}.wav")
            cmd = [
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                "-i", audio_path, "-ar", str(SR), "-ac", "1", "-f", "wav",
                part_path,
            ]
            run_cmd(cmd, check=False)
            if os.path.exists(part_path) and os.path.getsize(part_path) > 1000:
                parts.append(part_path)

        # concat
        list_file = os.path.join(tmpdir, "concat.txt")
        with open(list_file, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-ar", str(SR), "-ac", "1", "-f", "wav", output_wav,
        ]
        run_cmd(cmd)

    total_sec = sum(e - s for s, e in merged)
    elapsed = time.time() - t0
    print(f"  提取完成: {len(merged)}段, 总时长 {total_sec:.0f}s, 耗时 {elapsed:.0f}s")
    return total_sec


# ═══════════════════════════════════════════════════════════════════════════════
# 主分析管线
# ═══════════════════════════════════════════════════════════════════════════════
def analyze(
    audio_path: str,
    asr_backend: ASRBackend,
    speakers: int = 2,
    target_speaker: Union[str, int] = "auto",
    output_path: Optional[str] = None,
    clips_dir: Optional[str] = None,
    top_clips: int = 3,
    session_id: Optional[str] = None,
    skip_asr: bool = False,
    asr_json: Optional[str] = None,
    skip_clips: bool = False,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """主分析管线：从音频到结构化 JSON 的全量分析。

    执行步骤:
    1. ASR + 说话人分离（通过 asr_backend）
    2. 目标说话人识别（auto 或指定）
    3. 加载原始音频 + 声学特征提取
    4. 发音清晰度分析
    5. 文本指标提取（口癖/语速/句长/停顿）
    6. 情绪曲线 + 疲劳检测
    7. 互动节奏 + 非语音事件
    8. 话题分割 + 切片候选
    9. 综合评分

    Args:
        audio_path: 音频文件路径。
        asr_backend: ASR 后端实例（ASRBackend 子类）。
        speakers: 预期说话人数量。
        target_speaker: 目标说话人，"auto" 自动识别或指定 ID。
        output_path: 输出 JSON 路径，None 则自动生成。
        clips_dir: 切片保存目录，None 则不切片。
        top_clips: 提取切片数量上限。
        session_id: 场次 ID，None 则从文件名推断。
        skip_asr: 是否跳过 ASR（使用已有 JSON）。
        asr_json: 已有的 ASR JSON 路径（配合 skip_asr 使用）。
        skip_clips: 是否跳过切片提取。
        config_path: 配置文件目录路径，None 则使用默认位置。

    Returns:
        分析结果字典，包含 meta, acoustic, text_metrics, emotion,
        interaction, non_speech_events, topics, clip_candidates, scores。
    """
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在 {audio_path}", file=sys.stderr)
        sys.exit(1)

    # session_id
    if session_id:
        sid = session_id
    else:
        sid = os.path.splitext(os.path.basename(audio_path))[0][:8]

    output_path = output_path or f"{sid}_analysis.json"
    clips_dir = clips_dir if not skip_clips else None

    print(f"{'='*60}")
    print(f"直播音频全量分析 | session={sid}")
    print(f"音频: {audio_path}")
    print(f"输出: {output_path}")
    print(f"切片: {clips_dir or '跳过'}")
    print(f"{'='*60}")

    total_t0 = time.time()

    # 预加载配置
    filler_config = load_filler_config(config_path)
    speaker_config = load_speaker_config(config_path)

    # ── Step 1: ASR ──
    work_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(work_dir, exist_ok=True)

    if skip_asr and asr_json:
        asr_path = asr_json
        print(f"[1/9] 跳过ASR，使用已有: {asr_path}")
    else:
        asr_path = os.path.join(work_dir, f"{sid}_diarization.json")
        if os.path.exists(asr_path):
            print(f"[1/9] ASR结果已存在，跳过: {asr_path}")
        else:
            run_asr(audio_path, speakers, asr_path, asr_backend)

    sentences, asr_duration = load_asr_json(asr_path)
    print(f"  加载: {len(sentences)} sentences, 总时长 {asr_duration:.0f}s")

    # ── Step 2: 识别目标说话人 ──
    if target_speaker == "auto":
        target_sp, speaker_stats = auto_detect_speaker(sentences, speaker_config)
    else:
        target_sp = int(target_speaker)
        speaker_stats = {}
    print(f"  目标说话人: speaker_id={target_sp}")

    # ── Step 3: 加载原始音频（用于时间段查询，保持原始时间轴） ──
    # 注意：必须用原始音频而非提取的speaker WAV，因为ASR时间戳对应原始时间轴
    print(f"[3/9] 加载原始音频: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    feats = compute_acoustic_features(y, sr)

    # 总体声学摘要
    f0_v = feats["f0_valid"]
    rms_all = feats["rms"]
    rms_nz = rms_all[rms_all > 0]
    acoustic_summary = {
        "pitch": {
            "mean_hz": safe_float(np.mean(f0_v)) if len(f0_v) > 0 else 0,
            "median_hz": safe_float(np.median(f0_v)) if len(f0_v) > 0 else 0,
            "std_hz": safe_float(np.std(f0_v)) if len(f0_v) > 0 else 0,
        },
        "energy": {
            "mean_rms": safe_float(np.mean(rms_nz)) if len(rms_nz) > 0 else 0,
            "dynamic_range_db": safe_float(
                float(librosa.amplitude_to_db(np.max(rms_nz)))
                - float(librosa.amplitude_to_db(np.percentile(rms_nz, 5)))
            ) if len(rms_nz) > 0 else 0,
            "cv": safe_float(np.std(rms_nz) / np.mean(rms_nz))
            if len(rms_nz) > 0 and np.mean(rms_nz) > 0
            else 0,
        },
        "spectral": {
            "centroid_hz": safe_float(np.mean(feats["spectral_centroid"])),
            "flatness_mean": safe_float(np.mean(feats["spectral_flatness"]), 5),
            "zcr": safe_float(np.mean(feats["zcr"]), 4),
        },
    }

    # ── Step 5: 发音清晰度 ──
    clarity = compute_clarity(sentences, target_sp, feats)
    acoustic_summary["clarity_score"] = clarity["clarity_score"]

    # ── Step 6: 文本指标 ──
    text_metrics = compute_text_metrics(sentences, target_sp, filler_config)

    # ── Step 7: 情绪 + 疲劳 ──
    emotion_result, emotion_curve = compute_emotion_and_fatigue(
        sentences, target_sp, feats
    )

    # ── Step 8: 互动 + 非语音 ──
    interaction = compute_interaction(sentences, speakers)
    non_speech = compute_non_speech_events(feats)

    # ── Step 9: 话题 + 切片 ──
    topics, clip_candidates = compute_topics_and_clips(
        sentences, target_sp, feats, audio_path, clips_dir, top_clips,
        filler_config,
    )

    # ── 综合评分 ──
    scores: Dict[str, float] = {}
    f0_std = acoustic_summary["pitch"]["std_hz"]
    scores["pitch_expressiveness"] = min(100, max(0, f0_std / 40 * 100))
    e_cv = acoustic_summary["energy"]["cv"]
    scores["energy_dynamics"] = min(100, max(0, e_cv / 0.5 * 100))
    rate = text_metrics["speech_rate"]["mean_cpm"]
    if 200 <= rate <= 350:
        scores["speech_rate"] = 100
    elif rate < 200:
        scores["speech_rate"] = max(0, rate / 200 * 100)
    else:
        scores["speech_rate"] = max(0, 100 - (rate - 350) / 2)
    frag = text_metrics["sentence_length"]["fragmentation_rate"]
    core_filler_d = text_metrics["filler_words"]["core_density_per_1000"]
    fluency = 100 - min(50, frag * 500) - min(50, max(0, core_filler_d - 15) * 3)
    scores["fluency"] = max(0, min(100, fluency))
    scores["clarity"] = clarity["clarity_score"]
    gap_med = text_metrics["pauses"]["median_gap_sec"]
    if 0.2 <= gap_med <= 1.0:
        scores["pause_control"] = 100
    elif gap_med > 2.0:
        scores["pause_control"] = max(0, 100 - (gap_med - 2.0) * 30)
    else:
        scores["pause_control"] = 70

    # 逐字稿场景警告：段落级时间戳会导致互动数据异常
    if skip_asr and asr_json:
        ds = text_metrics.get("meta", {}).get("data_source", "")
        if ds in ("transcript_filtered", "transcript_partial"):
            print(
                "  ⚠ 逐字稿模式下停顿控制和互动节奏数据不可靠"
                "（段落间间隔含对方说话时间），已标注"
            )
            if "note" not in interaction:
                interaction["note"] = ""
            interaction["note"] += " [逐字稿模式: 数据不可靠，段落间间隔含对方说话时间]"
            scores["pause_control"] = 0
            scores["pause_control_note"] = "逐字稿模式不可用"

    weights = {
        "pitch_expressiveness": 0.20,
        "energy_dynamics": 0.20,
        "speech_rate": 0.20,
        "fluency": 0.15,
        "clarity": 0.15,
        "pause_control": 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    scores["overall"] = safe_float(overall)

    # ── 组装输出 JSON ──
    result: Dict[str, Any] = {
        "meta": {
            "session_id": sid,
            "audio_file": audio_path,
            "duration_seconds": safe_float(asr_duration),
            "speaker_count": speakers,
            "target_speaker": target_sp,
            "target_speaking_seconds": safe_float(
                text_metrics["meta"]["speaking_seconds"]
            ),
            "target_speaking_ratio": safe_float(
                text_metrics["meta"]["speaking_seconds"] / asr_duration
            ) if asr_duration > 0 else 0,
            "data_source": text_metrics["meta"].get("data_source", "asr"),
            "transcript_warning": text_metrics["meta"].get(
                "transcript_warning", ""
            ),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_duration_sec": safe_float(time.time() - total_t0),
        },
        "acoustic": acoustic_summary,
        "text_metrics": text_metrics,
        "emotion": emotion_result,
        "interaction": interaction,
        "non_speech_events": non_speech,
        "topics": topics,
        "clip_candidates": clip_candidates[:10],
        "scores": {k: safe_float(v) for k, v in scores.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"分析完成!")
    print(f"总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"输出: {output_path}")
    print(f"综合评分: {scores['overall']:.1f}/100")
    print(f"  音高表现力: {scores['pitch_expressiveness']:.0f}")
    print(f"  音量动态:   {scores['energy_dynamics']:.0f}")
    print(f"  语速合理性: {scores['speech_rate']:.0f}")
    print(f"  流畅度:     {scores['fluency']:.0f}")
    print(f"  发音清晰度: {scores['clarity']:.0f}")
    print(f"  停顿控制:   {scores['pause_control']:.0f}")
    print(f"{'='*60}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """argparse 命令行入口，解析参数后调用 analyze() 主管线。"""
    parser = argparse.ArgumentParser(description="直播音频全量分析")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    parser.add_argument(
        "--speakers", type=int, default=2, help="说话人数量"
    )
    parser.add_argument(
        "--target-speaker", default="auto", help="目标说话人ID (auto/0/1)"
    )
    parser.add_argument(
        "--output", default=None,
        help="输出JSON路径 (默认: {session_id}_analysis.json)",
    )
    parser.add_argument(
        "--clips-dir", default=None, help="切片保存目录 (不指定则不切片)"
    )
    parser.add_argument(
        "--top-clips", type=int, default=3, help="提取切片数量"
    )
    parser.add_argument(
        "--session-id", default=None, help="场次ID (如 0601)"
    )
    parser.add_argument(
        "--skip-asr", action="store_true", help="跳过ASR，使用已有JSON"
    )
    parser.add_argument(
        "--asr-json", default=None, help="已有的ASR JSON路径"
    )
    parser.add_argument(
        "--skip-clips", action="store_true", help="跳过切片提取"
    )
    parser.add_argument(
        "--config-dir", default=None, help="配置文件目录路径"
    )
    args = parser.parse_args()

    # 延迟导入 librosa（首次 import 较慢）
    print("[0/9] Importing librosa...")
    # librosa 已在模块顶部导入，这里仅作进度提示

    # 创建默认 ASR 后端
    from asr.bl_backend import BLASRBackend

    asr_backend = BLASRBackend()

    analyze(
        audio_path=args.audio,
        asr_backend=asr_backend,
        speakers=args.speakers,
        target_speaker=args.target_speaker,
        output_path=args.output,
        clips_dir=args.clips_dir,
        top_clips=args.top_clips,
        session_id=args.session_id,
        skip_asr=args.skip_asr,
        asr_json=args.asr_json,
        skip_clips=args.skip_clips,
        config_path=args.config_dir,
    )


if __name__ == "__main__":
    main()
