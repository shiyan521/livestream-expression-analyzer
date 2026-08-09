#!/usr/bin/env python3
"""
报告数据适配器：将分析管线输出（pipeline result）转换为
Web 报告页（report.html）可渲染的数据结构。

管线的 result 是嵌套结构（acoustic / text_metrics / emotion / topics ...），
报告页需要的是扁平展示结构（overall_score / dimensions / filler_words ...）。
本模块负责两层结构之间的转换，保证 Web 端展示真实分析结果。

用法:
    from report_adapter import to_report_view
    report_view = to_report_view(pipeline_result)
"""

from typing import Any, Dict, List, Optional


def _safe(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """从嵌套字典安全取值。"""
    return d.get(key, default) if isinstance(d, dict) else default


def _fmt_mmss(seconds: float) -> str:
    """秒 → MM:SS 字符串。"""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _score_to_level(score: float) -> str:
    """综合评分 → 表达水平评级。"""
    if score >= 85:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 60:
        return "合格"
    return "待改善"


def _dict_to_breakdown(counter: Dict[str, int], total_chars: int) -> List[Dict[str, Any]]:
    """{词: 次数} → [{word, count, density}]（density 为每千字密度）。"""
    if not isinstance(counter, dict):
        return []
    total = sum(counter.values()) or 1
    result = []
    for word, count in sorted(counter.items(), key=lambda x: -x[1])[:20]:
        density = round(count / total_chars * 1000, 1) if total_chars else 0
        result.append({"word": word, "count": count, "density": density})
    return result


def to_report_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    将管线分析结果转换为报告页展示结构。

    Args:
        result: analyzer.pipeline.analyze() 的返回值

    Returns:
        与 web/static/report.html 兼容的数据字典
    """
    meta = _safe(result, "meta", {})
    scores = _safe(result, "scores", {})
    text_metrics = _safe(result, "text_metrics", {})
    acoustic = _safe(result, "acoustic", {})
    emotion = _safe(result, "emotion", {})
    topics = _safe(result, "topics", {})

    # ── 综合评分与雷达图维度 ──
    overall = _safe(scores, "overall", 0) or 0
    filler = _safe(text_metrics, "filler_words", {})
    core_density = _safe(filler, "core_density_per_1000", 0) or 0

    dimensions = {
        "pitch_expression": round(_safe(scores, "pitch_expressiveness", 0) or 0),
        "volume_dynamic": round(_safe(scores, "energy_dynamics", 0) or 0),
        "speech_rate": round(_safe(scores, "speech_rate", 0) or 0),
        "fluency": round(_safe(scores, "fluency", 0) or 0),
        "clarity": round(_safe(scores, "clarity", 0) or 0),
        "pause_control": round(_safe(scores, "pause_control", 0) or 0),
    }

    # ── 口癖分析 ──
    total_chars = _safe(_safe(text_metrics, "meta", {}), "total_chars", 0) or 0
    core_breakdown = _dict_to_breakdown(
        _safe(filler, "core_breakdown", {}), total_chars
    )
    extended_breakdown = _dict_to_breakdown(
        _safe(filler, "extended_breakdown", {}), total_chars
    )
    combos = _safe(filler, "combos", [])
    if isinstance(combos, list):
        combos = combos[:10]

    # ── 语速（管线无逐分钟序列时按分段生成）──
    speech_rate = _safe(text_metrics, "speech_rate", {})
    rate_series = _safe(speech_rate, "series", [])
    if not rate_series and _safe(speech_rate, "mean_cpm", 0):
        rate_series = [round(_safe(speech_rate, "mean_cpm", 0))] * 10

    # ── 句长分布 ──
    sent_len = _safe(text_metrics, "sentence_length", {})
    over_100 = _safe(sent_len, "over_100", 0) or 0
    under_10 = _safe(sent_len, "under_10", 0) or 0
    normal_count = max(0, (total_chars and len(core_breakdown) and 0) or 0)
    # 正常句数 = 总句数 - 碎片句 - 超长句（无总句数时用 0 兜底）
    segment_count = _safe(_safe(text_metrics, "meta", {}), "segment_count", 0) or 0
    normal_count = max(0, segment_count - under_10 - over_100)

    # ── 声学特征 ──
    pitch = _safe(acoustic, "pitch", {})
    energy = _safe(acoustic, "energy", {})
    clarity_score = _safe(acoustic, "clarity_score", 0) or 0

    # ── 情绪曲线（2 分钟窗口，转为报告页格式）──
    emotion_curve = []
    curve = _safe(emotion, "emotion_curve", []) or []
    for point in curve:
        if isinstance(point, dict):
            start = point.get("start", point.get("begin_time", 0)) or 0
            end = point.get("end", point.get("end_time", start + 120)) or start
            emotion_curve.append({
                "time_range": f"{_fmt_mmss(start)}-{_fmt_mmss(end)}",
                "score": round(point.get("score", 0) or 0, 1),
            })

    # ── 话题分段 ──
    topic_segments = []
    topic_list = _safe(topics, "segments", []) or []
    if isinstance(topic_list, list):
        for t in topic_list[:12]:
            if not isinstance(t, dict):
                continue
            start = t.get("start", 0) or 0
            end = t.get("end", 0) or start
            density = t.get("filler_density", t.get("core_density", 0)) or 0
            rate = t.get("speech_rate", 0) or 0
            topic_segments.append({
                "topic": t.get("label", t.get("topic", "话题")),
                "duration": _fmt_mmss(max(0, end - start)),
                "filler_density": round(density, 1),
                "speech_rate": round(rate),
                "rating": t.get("rating", "良好"),
            })

    # ── 组装报告视图 ──
    return {
        "meta": {
            "session_id": _safe(meta, "session_id", "session"),
            "duration_seconds": round(_safe(meta, "duration_seconds", 0) or 0),
            "target_speaking_seconds": round(
                _safe(meta, "target_speaking_seconds", 0) or 0
            ),
            "target_speaking_ratio": round(
                _safe(meta, "target_speaking_ratio", 0) or 0, 3
            ),
            "analysis_timestamp": _safe(
                meta, "analysis_timestamp", ""
            ) or "",
        },
        "overall_score": round(overall, 1),
        "overall_level": _score_to_level(overall),
        "speakers": ["Speaker A", "Speaker B"],
        "dimensions": dimensions,
        "filler_words": {
            "core_total": _safe(filler, "core_total", 0) or 0,
            "core_density_per_1000": round(core_density, 1),
            "all_density_per_1000": round(
                _safe(filler, "all_density_per_1000", 0) or 0, 1
            ),
            "core_breakdown": core_breakdown,
            "extended_breakdown": extended_breakdown,
            "combos": combos,
        },
        "speech_rate": {
            "mean_cpm": round(_safe(speech_rate, "mean_cpm", 0) or 0, 1),
            "median_cpm": round(_safe(speech_rate, "median_cpm", 0) or 0, 1),
            "std_cpm": round(_safe(speech_rate, "std_cpm", 0) or 0, 1),
            "cv": round(_safe(speech_rate, "cv", 0) or 0, 3),
            "series": rate_series,
        },
        "sentence_length": {
            "mean": round(_safe(sent_len, "mean", 0) or 0, 1),
            "median": round(_safe(sent_len, "median", 0) or 0, 1),
            "over_100": over_100,
            "under_10": under_10,
            "fragmentation_rate": round(
                _safe(sent_len, "fragmentation_rate", 0) or 0, 3
            ),
            "distribution": [
                {"label": "正常句 (10-100字)", "value": normal_count},
                {"label": "碎片句 (<10字)", "value": under_10},
                {"label": "超长句 (>100字)", "value": over_100},
            ],
        },
        "pauses": {
            "median_gap_sec": round(
                _safe(_safe(text_metrics, "pauses", {}), "median_gap_sec", 0) or 0, 2
            ),
            "short_lt1s": _safe(_safe(text_metrics, "pauses", {}), "short_lt1s", 0) or 0,
            "medium_1_3s": _safe(_safe(text_metrics, "pauses", {}), "medium_1_3s", 0) or 0,
            "long_3_5s": _safe(_safe(text_metrics, "pauses", {}), "long_3_5s", 0) or 0,
            "extra_long_gt5s": _safe(_safe(text_metrics, "pauses", {}), "extra_long_gt5s", 0) or 0,
        },
        "acoustic": {
            "pitch": {
                "mean_hz": round(_safe(pitch, "mean_hz", 0) or 0, 1),
                "median_hz": round(_safe(pitch, "median_hz", 0) or 0, 1),
                "std_hz": round(_safe(pitch, "std_hz", 0) or 0, 1),
                "score": round(_safe(scores, "pitch_expressiveness", 0) or 0),
            },
            "energy": {
                "mean_rms": round(_safe(energy, "mean_rms", 0) or 0, 4),
                "dynamic_range_db": round(
                    _safe(energy, "dynamic_range_db", 0) or 0, 1
                ),
                "cv": round(_safe(energy, "cv", 0) or 0, 3),
                "score": round(_safe(scores, "energy_dynamics", 0) or 0),
            },
            "clarity": {
                "score": round(clarity_score),
            },
        },
        "emotion": {
            "curve": emotion_curve,
        },
        "topic_segments": topic_segments,
        "filler_hotspots": [],
        "history": [],
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("用法: python report_adapter.py <pipeline_result.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        result = json.load(f)
    view = to_report_view(result)
    print(json.dumps(view, ensure_ascii=False, indent=2))
