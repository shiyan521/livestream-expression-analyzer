"""
内容层指标模块。

基于 dialogue_parser 的对话结构分析结果，计算内容层面的核心指标:
  - 对话接球率（Catch Rate）: 目标说话人有效回应嘉宾内容的比例
  - 信息增量密度（Information Increment Density）: 每次回应新增信息的比例
  - 内容深度（Content Depth）: 核心观点数量、论据类型分布、新颖度
  - 话题连贯性（Topic Coherence）: 回应与问题的关联度、话题漂移检测

分析策略:
  - 规则优先: 字符 n-gram 重叠度、词汇新颖度等可纯规则计算
  - LLM 增强: 核心观点提取、论据类型判断可选用 LLM 提升精度
  - 时间分段: 所有指标按时间分段计算，观察趋势变化

用法:
  from analyzer.content_metrics import compute_content_metrics
  from analyzer.dialogue_parser import parse_dialogue

  dialogue = parse_dialogue(sentences, target_speaker=0)
  content = compute_content_metrics(sentences, target_speaker=0, dialogue_result=dialogue)
  print(content["summary"]["content_quality_score"])
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from analyzer.utils import fmt_time, safe_float


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 时间分段数量（将会话均分为 N 段）
TIME_SEGMENTS = 5

# 信息增量阈值
LOW_INCREMENT_THRESHOLD = 0.20   # 低于此值视为低增量回应
HIGH_INCREMENT_THRESHOLD = 0.60  # 高于此值视为高增量回应

# 接球率分段阈值
LOW_CATCH_THRESHOLD = 0.40

# 话题漂移检测：连续 shift 超过此次则视为漂移
TOPIC_DRIFT_CONSECUTIVE = 2

# 最低回应长度（字符），低于此值的回应不参与信息增量计算
MIN_CHARS_FOR_INCREMENT = 10

# n-gram 大小
NGRAM_SIZE = 2


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def _get_ngrams(text: str, n: int = NGRAM_SIZE) -> set:
    """提取字符级 n-gram 集合。

    Args:
        text: 输入文本。
        n: n-gram 大小。

    Returns:
        n-gram 字符串集合。
    """
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _time_segment_label(index: int, total: int, duration: float) -> str:
    """生成时间分段标签。

    Args:
        index: 分段索引（0-based）。
        total: 总分段数。
        duration: 总时长（秒）。

    Returns:
        标签字符串，如 "0-24min"。
    """
    seg_dur = duration / total if total > 0 else 0
    start_min = index * seg_dur / 60
    end_min = (index + 1) * seg_dur / 60
    return f"{start_min:.0f}-{end_min:.0f}min"


def _compute_trend(values: List[float]) -> str:
    """判断指标趋势。

    将分段值分为前半段和后半段，比较均值判断趋势。

    Args:
        values: 按时间顺序排列的指标值列表。

    Returns:
        "improving" / "declining" / "stable"
    """
    if len(values) < 2:
        return "stable"

    mid = len(values) // 2
    first_half = values[:mid] if mid > 0 else values[:1]
    second_half = values[mid:] if mid > 0 else values[1:]

    avg_first = sum(first_half) / len(first_half) if first_half else 0
    avg_second = sum(second_half) / len(second_half) if second_half else 0

    diff = avg_second - avg_first
    # 变化幅度小于 5% 视为稳定
    if abs(diff) < 0.05 or (avg_first > 0 and abs(diff) / avg_first < 0.05):
        return "stable"
    return "improving" if diff > 0 else "declining"


# ══════════════════════════════════════════════════════════════
# 1. 对话接球率分析
# ══════════════════════════════════════════════════════════════
def analyze_catch_rate(
    exchanges: List[Dict[str, Any]],
    duration: float,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """分析对话接球率，含时间分段和趋势。

    Args:
        exchanges: dialogue_parser 输出的交换对列表。
        duration: 会话总时长（秒）。
        num_segments: 时间分段数量。

    Returns:
        接球率分析结果:
            overall_catch_rate: 总体接球率
            total_exchanges: 交换对总数
            time_segments: 各时间段的接球率
            trend: 趋势 (improving/declining/stable)
            low_catch_exchanges: 接球失败的交换对列表
    """
    if not exchanges:
        return {
            "overall_catch_rate": 0.0,
            "total_exchanges": 0,
            "time_segments": [],
            "trend": "stable",
            "low_catch_exchanges": [],
        }

    # 总体接球率
    extend_count = sum(
        1 for ex in exchanges if ex.get("response_type") == "extend"
    )
    overall_rate = safe_float(extend_count / len(exchanges))

    # 时间分段
    seg_dur = duration / num_segments if duration > 0 else 0
    segments: List[Dict[str, Any]] = []
    seg_rates: List[float] = []

    for i in range(num_segments):
        seg_start = i * seg_dur
        seg_end = (i + 1) * seg_dur
        seg_exchanges = [
            ex for ex in exchanges
            if seg_start <= ex.get("question_time", 0) < seg_end
        ]
        if seg_exchanges:
            seg_extend = sum(
                1 for ex in seg_exchanges if ex.get("response_type") == "extend"
            )
            seg_rate = safe_float(seg_extend / len(seg_exchanges))
        else:
            seg_rate = 0.0
        seg_rates.append(seg_rate)
        segments.append({
            "segment": _time_segment_label(i, num_segments, duration),
            "catch_rate": seg_rate,
            "exchanges": len(seg_exchanges),
        })

    trend = _compute_trend(seg_rates)

    # 接球失败的交换对（非 extend 类型）
    low_catch = [
        {
            "time": ex.get("question_time_str", ""),
            "type": ex.get("response_type", ""),
            "question_preview": ex.get("question_preview", ""),
            "response_preview": ex.get("response_preview", ""),
        }
        for ex in exchanges
        if ex.get("response_type") != "extend"
    ]

    return {
        "overall_catch_rate": overall_rate,
        "total_exchanges": len(exchanges),
        "time_segments": segments,
        "trend": trend,
        "low_catch_exchanges": low_catch[:20],
        "low_catch_count": len(low_catch),
    }


# ══════════════════════════════════════════════════════════════
# 2. 信息增量密度
# ══════════════════════════════════════════════════════════════
def compute_information_increment(
    exchanges: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    duration: float,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """计算信息增量密度。

    对每个交换对的回应文本，计算其相对于之前所有内容的 n-gram 新颖度。
    新颖度 = 回应中未出现在之前内容中的 n-gram 占比。

    Args:
        exchanges: 交换对列表。
        sentences: ASR 句子列表（用于构建历史内容基线）。
        target_speaker: 目标说话人 ID。
        duration: 会话总时长（秒）。
        num_segments: 时间分段数量。

    Returns:
        信息增量密度分析结果:
            avg_increment_ratio: 平均信息增量比例
            time_segments: 各时间段的增量密度
            trend: 趋势
            low_density_exchanges: 低增量回应列表
            high_density_exchanges: 高增量回应列表
    """
    if not exchanges:
        return {
            "avg_increment_ratio": 0.0,
            "time_segments": [],
            "trend": "stable",
            "low_density_exchanges": [],
            "high_density_exchanges": [],
        }

    # 构建累积 n-gram 历史
    cumulative_ngrams: set = set()
    increment_ratios: List[float] = []
    exchange_details: List[Dict[str, Any]] = []

    # 按时间排序 exchanges（确保按顺序处理）
    sorted_exchanges = sorted(exchanges, key=lambda x: x.get("question_time", 0))

    for ex in sorted_exchanges:
        resp_text = ex["response"]["text"].strip() if isinstance(ex.get("response"), dict) else ""
        ques_text = ex["question"]["text"].strip() if isinstance(ex.get("question"), dict) else ""

        if len(resp_text) < MIN_CHARS_FOR_INCREMENT:
            increment_ratios.append(0.0)
            exchange_details.append({
                "time": ex.get("question_time_str", ""),
                "increment_ratio": 0.0,
                "level": "skipped",
                "response_preview": resp_text[:60],
            })
            # 仍然将问题内容加入历史
            cumulative_ngrams.update(_get_ngrams(ques_text))
            continue

        resp_ngrams = _get_ngrams(resp_text)
        if not resp_ngrams:
            increment_ratios.append(0.0)
            continue

        # 新 n-gram 比例
        new_ngrams = resp_ngrams - cumulative_ngrams
        ratio = len(new_ngrams) / len(resp_ngrams) if resp_ngrams else 0
        increment_ratios.append(safe_float(ratio))

        # 分级
        if ratio < LOW_INCREMENT_THRESHOLD:
            level = "low"
        elif ratio > HIGH_INCREMENT_THRESHOLD:
            level = "high"
        else:
            level = "normal"

        exchange_details.append({
            "time": ex.get("question_time_str", ""),
            "increment_ratio": safe_float(ratio),
            "level": level,
            "response_preview": resp_text[:60],
        })

        # 将当前回应和问题加入历史
        cumulative_ngrams.update(resp_ngrams)
        cumulative_ngrams.update(_get_ngrams(ques_text))

    # 整体统计
    valid_ratios = [r for r in increment_ratios if r > 0]
    avg_ratio = safe_float(
        sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0
    )

    # 时间分段
    seg_dur = duration / num_segments if duration > 0 else 0
    segments: List[Dict[str, Any]] = []
    seg_avgs: List[float] = []

    for i in range(num_segments):
        seg_start = i * seg_dur
        seg_end = (i + 1) * seg_dur
        seg_details = [
            (ed, ex) for ed, ex in zip(exchange_details, sorted_exchanges)
            if seg_start <= ex.get("question_time", 0) < seg_end
        ]
        if seg_details:
            seg_ratios = [ed["increment_ratio"] for ed, _ in seg_details if ed["increment_ratio"] > 0]
            seg_avg = safe_float(sum(seg_ratios) / len(seg_ratios)) if seg_ratios else 0
        else:
            seg_avg = 0.0
        seg_avgs.append(seg_avg)
        segments.append({
            "segment": _time_segment_label(i, num_segments, duration),
            "avg_increment": seg_avg,
            "exchanges": len(seg_details),
        })

    trend = _compute_trend(seg_avgs)

    # 低增量和高增量交换对
    low_density = [ed for ed in exchange_details if ed["level"] == "low"]
    high_density = [ed for ed in exchange_details if ed["level"] == "high"]

    return {
        "avg_increment_ratio": avg_ratio,
        "time_segments": segments,
        "trend": trend,
        "low_density_exchanges": low_density[:15],
        "high_density_exchanges": high_density[:10],
        "low_density_count": len(low_density),
        "high_density_count": len(high_density),
    }


# ══════════════════════════════════════════════════════════════
# 3. 内容深度分析
# ══════════════════════════════════════════════════════════════
def analyze_content_depth(
    exchanges: List[Dict[str, Any]],
    target_speaker: int,
    llm_backend: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """分析内容深度: 核心观点提取、论据类型分布、新颖度。

    Args:
        exchanges: 交换对列表。
        target_speaker: 目标说话人 ID。
        llm_backend: 可选的 LLM 后端实例。
        config: 可选的分析配置。

    Returns:
        内容深度分析结果:
            total_core_points: 核心观点总数
            novel_points: 新颖观点数
            repeated_points: 重复观点数
            evidence_type_distribution: 论据类型分布
            avg_novelty_score: 平均新颖度
            core_points: 核心观点列表
    """
    # 收集所有目标说话人的回应文本
    responses: List[str] = []
    for ex in exchanges:
        resp = ex.get("response", {})
        if isinstance(resp, dict):
            text = resp.get("text", "").strip()
            if len(text) >= MIN_CHARS_FOR_INCREMENT:
                responses.append(text)

    if not responses:
        return {
            "total_core_points": 0,
            "novel_points": 0,
            "repeated_points": 0,
            "evidence_type_distribution": {},
            "avg_novelty_score": 0.0,
            "core_points": [],
        }

    # 核心观点提取
    core_points: List[Dict[str, Any]] = []
    evidence_types: List[str] = []

    # 尝试使用 LLM
    use_llm = False
    content_analyzer = None
    if llm_backend is not None:
        try:
            from analyzer.llm_analyzer import ContentAnalyzer
            content_analyzer = ContentAnalyzer(llm_backend=llm_backend, config=config)
            use_llm = not content_analyzer.is_mock
        except Exception:
            use_llm = False

    if use_llm and content_analyzer is not None:
        # LLM 模式：提取核心观点和论据类型
        for resp_text in responses:
            # 每次处理最多 2000 字符
            text_chunk = resp_text[:2000]
            points = content_analyzer.extract_core_points(text_chunk)
            for p in points:
                core_points.append({
                    "point": p.get("point", ""),
                    "novelty": safe_float(p.get("novelty", 0.5)),
                    "evidence_type": p.get("evidence_type", "none"),
                })

            # 论据类型
            etype = content_analyzer.detect_evidence_type(text_chunk)
            evidence_types.append(etype)
    else:
        # 规则模式：从文本中提取关键短语
        from analyzer.dialogue_parser import extract_key_phrases

        # 规则版论据类型检测
        def _detect_evidence_rule(text: str) -> str:
            if any(w in text for w in ["%", "亿", "万", "数据", "统计", "调查"]):
                return "data"
            if any(w in text for w in ["比如", "举个例子", "案例", "曾经"]):
                return "case"
            if any(w in text for w in ["就像", "好比", "相当于", "类似于"]):
                return "analogy"
            if any(w in text for w in ["引用", "说过", "提到", "认为"]):
                return "quote"
            if any(w in text for w in ["我", "我自己", "我的", "我们"]):
                return "personal_experience"
            return "none"

        seen_phrases: set = set()
        for resp_text in responses:
            phrases = extract_key_phrases(resp_text, max_phrases=3)
            for p in phrases:
                # 判断是否重复
                p_ngrams = _get_ngrams(p)
                is_repeat = False
                for seen in seen_phrases:
                    s_ngrams = _get_ngrams(seen)
                    if s_ngrams and p_ngrams:
                        overlap = len(p_ngrams & s_ngrams) / len(p_ngrams | s_ngrams)
                        if overlap > 0.5:
                            is_repeat = True
                            break

                core_points.append({
                    "point": p,
                    "novelty": 0.3 if is_repeat else 0.6,
                    "evidence_type": _detect_evidence_rule(resp_text),
                    "is_repeat": is_repeat,
                })
                seen_phrases.add(p)

            evidence_types.append(_detect_evidence_rule(resp_text))

    # 统计
    total_points = len(core_points)
    novel_points = sum(1 for p in core_points if p.get("novelty", 0) >= 0.5)
    repeated_points = total_points - novel_points
    avg_novelty = safe_float(
        sum(p.get("novelty", 0) for p in core_points) / total_points
    ) if total_points > 0 else 0

    evidence_dist = dict(Counter(evidence_types))

    return {
        "total_core_points": total_points,
        "novel_points": novel_points,
        "repeated_points": repeated_points,
        "evidence_type_distribution": evidence_dist,
        "avg_novelty_score": avg_novelty,
        "core_points": core_points[:30],
        "analysis_mode": "llm" if use_llm else "rule",
    }


# ══════════════════════════════════════════════════════════════
# 4. 话题连贯性
# ══════════════════════════════════════════════════════════════
def analyze_topic_coherence(
    exchanges: List[Dict[str, Any]],
    duration: float,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """分析话题连贯性: 回应与问题的关联度、话题漂移检测。

    Args:
        exchanges: 交换对列表。
        duration: 会话总时长（秒）。
        num_segments: 时间分段数量。

    Returns:
        话题连贯性分析结果:
            avg_relevance: 平均关联度
            topic_drift_count: 话题漂移次数
            coherence_trend: 连贯性趋势
            drift_exchanges: 漂移交换对列表
    """
    if not exchanges:
        return {
            "avg_relevance": 0.0,
            "topic_drift_count": 0,
            "coherence_trend": "stable",
            "drift_exchanges": [],
        }

    # 计算每个交换对的关联度（问题与回应的 n-gram 相似度）
    relevance_scores: List[float] = []
    sorted_exchanges = sorted(exchanges, key=lambda x: x.get("question_time", 0))

    for ex in sorted_exchanges:
        ques_text = ex["question"]["text"].strip() if isinstance(ex.get("question"), dict) else ""
        resp_text = ex["response"]["text"].strip() if isinstance(ex.get("response"), dict) else ""

        if not ques_text or not resp_text:
            relevance_scores.append(0.0)
            continue

        ques_ngrams = _get_ngrams(ques_text)
        resp_ngrams = _get_ngrams(resp_text)

        if not ques_ngrams or not resp_ngrams:
            relevance_scores.append(0.0)
            continue

        # Jaccard 相似度作为关联度
        intersection = ques_ngrams & resp_ngrams
        union = ques_ngrams | resp_ngrams
        relevance = len(intersection) / len(union) if union else 0
        relevance_scores.append(safe_float(relevance))

    avg_relevance = safe_float(
        sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    )

    # 话题漂移检测：连续 N 个 shift 类型
    drift_count = 0
    drift_exchanges: List[Dict[str, Any]] = []
    consecutive_shift = 0

    for i, ex in enumerate(sorted_exchanges):
        if ex.get("response_type") == "shift":
            consecutive_shift += 1
            if consecutive_shift >= TOPIC_DRIFT_CONSECUTIVE:
                drift_count += 1
                drift_exchanges.append({
                    "time": ex.get("question_time_str", ""),
                    "question_preview": ex.get("question_preview", ""),
                    "response_preview": ex.get("response_preview", ""),
                    "consecutive_shifts": consecutive_shift,
                })
        else:
            consecutive_shift = 0

    # 时间分段关联度趋势
    seg_dur = duration / num_segments if duration > 0 else 0
    seg_relevances: List[float] = []

    for i in range(num_segments):
        seg_start = i * seg_dur
        seg_end = (i + 1) * seg_dur
        seg_scores = [
            score for score, ex in zip(relevance_scores, sorted_exchanges)
            if seg_start <= ex.get("question_time", 0) < seg_end
        ]
        seg_avg = safe_float(sum(seg_scores) / len(seg_scores)) if seg_scores else 0
        seg_relevances.append(seg_avg)

    coherence_trend = _compute_trend(seg_relevances)

    return {
        "avg_relevance": avg_relevance,
        "topic_drift_count": drift_count,
        "coherence_trend": coherence_trend,
        "drift_exchanges": drift_exchanges[:10],
    }


# ══════════════════════════════════════════════════════════════
# 5. 综合评分
# ══════════════════════════════════════════════════════════════
def compute_content_score(
    catch_rate_result: Dict[str, Any],
    increment_result: Dict[str, Any],
    depth_result: Dict[str, Any],
    coherence_result: Dict[str, Any],
) -> Dict[str, float]:
    """计算内容层综合评分。

    评分维度和权重:
      - 接球率 (30%): extend 比例直接映射到 0-100
      - 信息增量 (30%): 平均增量比例映射到 0-100
      - 内容深度 (20%): 新颖观点比例 × 平均新颖度
      - 话题连贯 (20%): 平均关联度映射 + 漂移惩罚

    Args:
        catch_rate_result: 接球率分析结果。
        increment_result: 信息增量分析结果。
        depth_result: 内容深度分析结果。
        coherence_result: 话题连贯性分析结果。

    Returns:
        评分字典:
            catch_rate_score: 接球率得分 [0-100]
            information_density_score: 信息增量得分 [0-100]
            content_depth_score: 内容深度得分 [0-100]
            topic_coherence_score: 话题连贯性得分 [0-100]
            content_quality_score: 内容质量综合得分 [0-100]
    """
    # 接球率得分
    catch_rate = catch_rate_result.get("overall_catch_rate", 0)
    catch_score = safe_float(catch_rate * 100)

    # 信息增量得分
    avg_increment = increment_result.get("avg_increment_ratio", 0)
    # 0.4 的增量比例视为满分基准
    density_score = safe_float(min(100, avg_increment / 0.4 * 100))

    # 内容深度得分
    total_points = depth_result.get("total_core_points", 0)
    novel_points = depth_result.get("novel_points", 0)
    avg_novelty = depth_result.get("avg_novelty_score", 0)
    novelty_ratio = safe_float(novel_points / total_points) if total_points > 0 else 0
    depth_score = safe_float(novelty_ratio * 50 + avg_novelty * 50)

    # 话题连贯性得分
    avg_relevance = coherence_result.get("avg_relevance", 0)
    drift_count = coherence_result.get("topic_drift_count", 0)
    # 关联度 0.15 视为满分基准
    coherence_base = safe_float(min(100, avg_relevance / 0.15 * 100))
    # 每次漂移扣 5 分
    coherence_score = safe_float(max(0, coherence_base - drift_count * 5))

    # 综合
    weights = {
        "catch_rate": 0.30,
        "information_density": 0.30,
        "content_depth": 0.20,
        "topic_coherence": 0.20,
    }
    overall = (
        catch_score * weights["catch_rate"]
        + density_score * weights["information_density"]
        + depth_score * weights["content_depth"]
        + coherence_score * weights["topic_coherence"]
    )

    return {
        "catch_rate_score": catch_score,
        "information_density_score": density_score,
        "content_depth_score": depth_score,
        "topic_coherence_score": coherence_score,
        "content_quality_score": safe_float(overall),
    }


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
def compute_content_metrics(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    dialogue_result: Optional[Dict[str, Any]] = None,
    duration: float = 0,
    llm_backend: Any = None,
    config: Optional[Dict[str, Any]] = None,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """内容层分析主入口。

    基于 dialogue_parser 的对话结构分析结果，计算内容层面的四大指标:
    1. 对话接球率
    2. 信息增量密度
    3. 内容深度
    4. 话题连贯性

    如果未提供 dialogue_result，会自动调用 parse_dialogue 生成。

    Args:
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        dialogue_result: dialogue_parser 的输出。为 None 则自动生成。
        duration: 会话总时长（秒）。为 0 则从 sentences 推断。
        llm_backend: 可选的 LLM 后端实例。
        config: 可选的分析配置。
        num_segments: 时间分段数量。

    Returns:
        内容层分析结果字典，包含:
            catch_rate_analysis: 接球率分析
            information_density: 信息增量密度
            content_depth: 内容深度
            topic_coherence: 话题连贯性
            summary: 综合评分
    """
    print("[content_metrics] 内容层指标分析...")

    # 如果没有 dialogue_result，自动生成
    if dialogue_result is None:
        from analyzer.dialogue_parser import parse_dialogue
        dialogue_result = parse_dialogue(
            sentences, target_speaker, llm_backend=llm_backend
        )

    exchanges = dialogue_result.get("exchanges", [])

    if not exchanges:
        print("  无交换对，跳过内容层分析")
        return {
            "catch_rate_analysis": {},
            "information_density": {},
            "content_depth": {},
            "topic_coherence": {},
            "summary": {
                "content_quality_score": 0.0,
                "note": "无交换对（独白场或单人发言）",
            },
        }

    # 推断时长
    if duration <= 0 and sentences:
        duration = max(s["end"] for s in sentences)

    # 1. 接球率分析
    print("  [1/4] 对话接球率...")
    catch_result = analyze_catch_rate(exchanges, duration, num_segments)
    print(
        f"    接球率: {catch_result['overall_catch_rate']:.1%}, "
        f"趋势: {catch_result['trend']}, "
        f"接球失败: {catch_result['low_catch_count']} 次"
    )

    # 2. 信息增量密度
    print("  [2/4] 信息增量密度...")
    increment_result = compute_information_increment(
        exchanges, sentences, target_speaker, duration, num_segments
    )
    print(
        f"    平均增量: {increment_result['avg_increment_ratio']:.1%}, "
        f"趋势: {increment_result['trend']}, "
        f"低增量: {increment_result['low_density_count']} 次"
    )

    # 3. 内容深度
    print("  [3/4] 内容深度...")
    depth_result = analyze_content_depth(
        exchanges, target_speaker, llm_backend, config
    )
    print(
        f"    核心观点: {depth_result['total_core_points']}, "
        f"新颖: {depth_result['novel_points']}, "
        f"平均新颖度: {depth_result['avg_novelty_score']:.2f}, "
        f"模式: {depth_result.get('analysis_mode', 'rule')}"
    )

    # 4. 话题连贯性
    print("  [4/4] 话题连贯性...")
    coherence_result = analyze_topic_coherence(exchanges, duration, num_segments)
    print(
        f"    平均关联度: {coherence_result['avg_relevance']:.2f}, "
        f"漂移: {coherence_result['topic_drift_count']} 次, "
        f"趋势: {coherence_result['coherence_trend']}"
    )

    # 5. 综合评分
    scores = compute_content_score(
        catch_result, increment_result, depth_result, coherence_result
    )
    print(
        f"  内容质量综合得分: {scores['content_quality_score']:.1f}/100"
    )

    return {
        "catch_rate_analysis": catch_result,
        "information_density": increment_result,
        "content_depth": depth_result,
        "topic_coherence": coherence_result,
        "summary": scores,
    }
