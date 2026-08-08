"""
话题分割与切片候选模块。

包含以下功能:
- 间隔 + 文本标记 + 说话人模式三重检测的话题边界识别
- 话题段构建（含主导说话人、声学特征、口癖密度、语速）
- 切片候选评分与排序
- 自动话题标签（从 config/topic_labels.yaml 读取规则）
- 内容结构分类（开场/收束/案例/观点/互动）
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analyzer.utils import (
    fmt_time,
    load_filler_config,
    load_topic_label_config,
    run_cmd,
    safe_float,
)


def auto_topic_label(
    text: str,
    idx: int,
    label_rules: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """根据关键词规则为话题段落自动打标签。

    规则从 config/topic_labels.yaml 加载，按顺序匹配，命中即返回对应标签。
    如果没有配置规则或均未命中，返回通用标签 "话题N"。

    Args:
        text: 话题段落的文本内容。
        idx: 话题序号（从 0 开始）。
        label_rules: 话题标签规则列表，每条规则结构为
            {"keywords": [...], "label": "..."}。
            若为 None 则从 config 加载。

    Returns:
        话题标签字符串。
    """
    if label_rules is None:
        label_rules = load_topic_label_config()

    t = text[:200]
    for rule in label_rules:
        keywords = rule.get("keywords", [])
        label = rule.get("label", "")
        if any(k in t for k in keywords):
            return label

    return f"话题{idx + 1}"


def classify_structure(
    text: str,
    seg_sents: List[Dict[str, Any]],
    target_speaker: int,
) -> str:
    """根据文本内容分类话题段落的结构类型。

    分类优先级: 总结/收束 > 引入/开场 > 案例/故事 > 观点阐述 > 互动/问答

    Args:
        text: 话题段落的文本内容。
        seg_sents: 该段落内的句子列表。
        target_speaker: 目标说话人 ID。

    Returns:
        结构类型标签字符串。
    """
    if any(k in text[:100] for k in ["收尾", "感谢", "结束", "今天就", "拜拜", "先到这里"]):
        return "总结/收束"
    if any(k in text[:100] for k in ["开场", "开始吧", "大家好", "欢迎", "简单开个场"]):
        return "引入/开场"
    story_kw = ["我当时", "我之前", "有一次", "发现", "结果", "然后他就", "翻车", "做了", "试过"]
    if sum(1 for k in story_kw if k in text) >= 3:
        return "案例/故事"
    opinion_kw = ["我觉得", "我认为", "我的观点", "我的感受", "在我看来", "本质上", "其实"]
    if sum(1 for k in opinion_kw if k in text) >= 2:
        return "观点阐述"
    speakers = set(s["speaker"] for s in seg_sents)
    if len(speakers) >= 2:
        return "互动/问答"
    return "观点阐述"


def compute_topics_and_clips(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    feats: Dict[str, Any],
    audio_path: str,
    clips_dir: Optional[str],
    top_n: int,
    filler_config: Optional[Dict[str, List[str]]] = None,
    label_rules: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """基于间隔 + 文本标记 + 说话人模式的话题分割，自动切片。

    话题边界检测策略（三重检测）:
    1. 间隔 > 5s
    2. 文本头部命中话题标记词
    3. 说话人切换 + 间隔 > 2s（需前 3 句为同一说话人）

    切片候选评分: 0.20*能量分 + 0.15*口癖分 + 结构加分 + 独立性加分 + 内容加分

    Args:
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        feats: 声学特征字典。
        audio_path: 原始音频文件路径（用于切片提取）。
        clips_dir: 切片保存目录，None 则不提取切片。
        top_n: 提取切片数量上限。
        filler_config: 口癖词库配置，若为 None 则从 config 加载。
        label_rules: 话题标签规则，若为 None 则从 config 加载。

    Returns:
        元组 (topics, clips):
            topics: 话题段列表
            clips: 切片候选列表（按质量分排序）
    """
    print("[9/9] 话题分割 + 切片候选...")

    # 加载配置
    if filler_config is None:
        filler_config = load_filler_config()
    topic_markers = filler_config.get("topic_markers", [])
    filler_core = filler_config.get("core_fillers", [])

    if label_rules is None:
        label_rules = load_topic_label_config()

    # --- 话题边界 ---
    boundaries = [0]
    for i in range(1, len(sentences)):
        prev, curr = sentences[i - 1], sentences[i]
        gap = curr["start"] - prev["end"]
        if gap > 5.0:
            boundaries.append(i)
            continue
        # 文本标记
        text_head = curr["text"][:20]
        if any(m in text_head for m in topic_markers):
            boundaries.append(i)
            continue
        # 说话人切换 + 间隔>2s
        if i >= 3:
            prev_sp = [sentences[j]["speaker"] for j in range(i - 3, i)]
            if len(set(prev_sp)) == 1 and curr["speaker"] != prev_sp[0] and gap > 2.0:
                boundaries.append(i)

    # 过滤过近边界 (<30s)
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        if sentences[b]["start"] - sentences[filtered[-1]]["start"] > 30:
            filtered.append(b)
    boundaries = filtered

    # --- 构建话题段 ---
    topics: List[Dict[str, Any]] = []
    for i in range(len(boundaries)):
        s_idx = boundaries[i]
        e_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
        seg_sents = sentences[s_idx:e_idx]
        if not seg_sents:
            continue
        seg_start = seg_sents[0]["start"]
        seg_end = seg_sents[-1]["end"]
        seg_text = "".join(s["text"] for s in seg_sents)
        seg_dur = seg_end - seg_start

        # 主导说话人
        sp_time: Dict[int, float] = defaultdict(float)
        for s in seg_sents:
            sp_time[s["speaker"]] += s["end"] - s["start"]
        total_sp = sum(sp_time.values())
        target_ratio = sp_time.get(target_speaker, 0) / total_sp if total_sp > 0 else 0.5

        # 自动标签
        label = auto_topic_label(seg_text, i, label_rules)

        # 内容结构
        structure = classify_structure(seg_text, seg_sents, target_speaker)

        # 声学特征（该段目标说话人）
        f0 = feats["f0"]
        f0_times = feats["f0_times"]
        rms = feats["rms"]
        rms_times = feats["rms_times"]

        target_sents_in_seg = [s for s in seg_sents if s["speaker"] == target_speaker]
        if target_sents_in_seg:
            t0 = min(s["start"] for s in target_sents_in_seg)
            t1 = max(s["end"] for s in target_sents_in_seg)
            mask_f0 = (f0_times >= t0) & (f0_times < t1)
            f0_seg = f0[mask_f0]
            f0_seg = f0_seg[~np.isnan(f0_seg)]
            f0_mean = safe_float(np.mean(f0_seg)) if len(f0_seg) > 0 else 0
            f0_std = safe_float(np.std(f0_seg)) if len(f0_seg) > 0 else 0

            mask_rms = (rms_times >= t0) & (rms_times < t1)
            rms_seg = rms[mask_rms]
            rms_mean = safe_float(np.mean(rms_seg)) if len(rms_seg) > 0 else 0
        else:
            f0_mean = f0_std = rms_mean = 0

        # 口癖密度（仅核心口癖）
        filler_cnt = 0
        char_cnt = 0
        for s in seg_sents:
            if s["speaker"] == target_speaker:
                char_cnt += len(s["text"])
                for fw in filler_core:
                    filler_cnt += s["text"].count(fw)
        filler_d = safe_float(filler_cnt / (char_cnt / 100)) if char_cnt > 0 else 0

        # 语速
        sp_chars = sum(len(s["text"]) for s in target_sents_in_seg)
        sp_time_s = sum(s["end"] - s["start"] for s in target_sents_in_seg)
        rate_cps = safe_float(sp_chars / sp_time_s) if sp_time_s > 0 else 0

        topics.append({
            "id": i + 1,
            "start_sec": safe_float(seg_start),
            "end_sec": safe_float(seg_end),
            "duration_sec": safe_float(seg_dur),
            "label": label,
            "structure": structure,
            "target_speaker_ratio": safe_float(target_ratio),
            "acoustic": {
                "f0_mean_hz": f0_mean,
                "f0_std_hz": f0_std,
                "rms_mean": rms_mean,
            },
            "filler_density": filler_d,
            "speech_rate_cps": rate_cps,
        })

    # --- 切片候选 ---
    avg_rms = np.mean([t["acoustic"]["rms_mean"] for t in topics]) if topics else 0.001
    avg_filler = np.mean([t["filler_density"] for t in topics]) if topics else 1.0

    clips: List[Dict[str, Any]] = []
    for t in topics:
        if t["duration_sec"] < 45 or t["duration_sec"] > 360:
            continue

        e_score = min(100, max(0, (t["acoustic"]["rms_mean"] / max(avg_rms, 0.001) - 0.5) * 80))
        f_score = min(100, max(0, (1 - t["filler_density"] / max(avg_filler, 0.01)) * 80))

        bonus = {
            "案例/故事": 25,
            "互动/问答": 20,
            "观点阐述": 10,
            "引入/开场": 5,
        }.get(t["structure"], 0)
        standalone = 10 if 60 <= t["duration_sec"] <= 300 else 0
        content = min(15, t["duration_sec"] / 30)

        q = e_score * 0.20 + f_score * 0.15 + bonus + standalone + content
        q = min(100, max(0, q))

        clips.append({
            "rank": 0,
            "topic_id": t["id"],
            "label": t["label"],
            "start_sec": t["start_sec"],
            "end_sec": t["end_sec"],
            "duration_sec": t["duration_sec"],
            "quality_score": safe_float(q),
            "structure": t["structure"],
        })

    clips.sort(key=lambda x: x["quality_score"], reverse=True)
    for i, c in enumerate(clips):
        c["rank"] = i + 1

    # --- 提取 top N 切片 ---
    extracted: List[str] = []
    if clips_dir and top_n > 0:
        os.makedirs(clips_dir, exist_ok=True)
        for c in clips[:top_n]:
            title = c["label"][:10].replace(" ", "_").replace("/", "_")
            out_file = os.path.join(clips_dir, f"clip_{c['rank']:02d}_{title}.mp3")
            start_adj = max(0, c["start_sec"] - 1)
            dur_adj = c["duration_sec"] + 2
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_adj:.2f}",
                "-i", audio_path,
                "-t", f"{dur_adj:.2f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                out_file,
            ]
            r = run_cmd(cmd, check=False)
            if r.returncode == 0 and os.path.exists(out_file):
                c["file"] = out_file
                extracted.append(out_file)
                print(f"  切片 {c['rank']}: {c['label']} -> {out_file}")

    print(f"  话题: {len(topics)}, 切片候选: {len(clips)}, 已提取: {len(extracted)}")
    return topics, clips
