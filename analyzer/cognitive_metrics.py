"""
认知层指标模块。

基于 content_metrics 的对话结构分析和 deep_analysis 的表达层指标，
计算认知层面的三大指标:
  1. 论证链追踪 (Argument Chain Tracking): 论证完整性、链条深度、逻辑跳跃检测
  2. 思维深度评估 (Thinking Depth Assessment): 表面/分析/综合/元认知四层分类
  3. 认知负荷与内容质量关联 (Cognitive-Content Correlation): 负荷-质量交叉分析

分析策略:
  - 规则优先: 逻辑连接词模式、抽象词比例、句式复杂度等可纯规则计算
  - LLM 增强: 论证结构提取、思维深度分类可选用 LLM 提升精度
  - 时间分段: 所有指标按时间分段计算，观察趋势变化

用法:
  from analyzer.cognitive_metrics import compute_cognitive_metrics
  from analyzer.dialogue_parser import parse_dialogue
  from analyzer.content_metrics import compute_content_metrics

  dialogue = parse_dialogue(sentences, target_speaker=0)
  content = compute_content_metrics(sentences, target_speaker=0, dialogue_result=dialogue)
  cognitive = compute_cognitive_metrics(
      sentences, target_speaker=0,
      dialogue_result=dialogue,
      content_result=content,
  )
  print(cognitive["summary"]["cognitive_quality_score"])
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from analyzer.utils import fmt_time, safe_float


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

TIME_SEGMENTS = 5

# ── 论证链相关 ──
# 因果连接词（标记论证推进）
CAUSAL_MARKERS = ["因为", "所以", "因此", "导致", "由于", "原因是", "之所以"]
# 递进连接词（标记论证深化）
PROGRESSIVE_MARKERS = ["而且", "另外", "同时", "并且", "不仅", "甚至", "更重要的是"]
# 总结连接词（标记论证收束）
SUMMARY_MARKERS = ["总之", "总的来说", "最终", "总结一下", "核心就是", "关键就是", "本质", "核心", "归根结底"]
# 举例连接词（标记论据展开）
EXAMPLE_MARKERS = ["比如", "例如", "举个例子", "比如说", "像是", "举个例子来说"]
# 转折连接词（标记论证方向调整）
TRANSITION_MARKERS = ["但是", "不过", "然而", "虽然", "尽管", "反而", "但是话说回来"]

# 论证结构标记
CLAIM_MARKERS = ["我觉得", "我认为", "我的判断", "我的理解", "我倾向于", "我比较看好", "本质上"]
EVIDENCE_MARKERS = EXAMPLE_MARKERS + ["数据", "统计", "调查", "报告", "研究", "实际上"]
CONCLUSION_MARKERS = SUMMARY_MARKERS + ["所以", "因此", "这就是为什么", "这就意味着"]

# 最低回应长度（字符）
MIN_CHARS_FOR_ANALYSIS = 10

# ── 思维深度相关 ──
# 四个思维深度层级
DEPTH_SURFACE = "surface"        # 现象描述：事实陈述、个人经历、场景描述
DEPTH_ANALYSIS = "analysis"      # 原因分析：因果推导、对比分析、逻辑拆解
DEPTH_SYNTHESIS = "synthesis"    # 综合判断：框架构建、规律提炼、本质洞察
DEPTH_META = "meta"              # 元认知：自我反思、认知校准、方法论

ALL_DEPTH_LEVELS = [DEPTH_SURFACE, DEPTH_ANALYSIS, DEPTH_SYNTHESIS, DEPTH_META]

# 深度评分权重
DEPTH_WEIGHTS = {
    DEPTH_SURFACE: 1,
    DEPTH_ANALYSIS: 2,
    DEPTH_SYNTHESIS: 3,
    DEPTH_META: 4,
}

# 表面层指标词（现象描述、个人经历）
SURFACE_INDICATORS = [
    "我看到", "我听到", "我发现", "我去", "我之前", "我遇到",
    "昨天", "今天", "上次", "那个时候", "当时",
    "人", "公司", "产品", "视频", "直播", "手机",
]
# 分析层指标词（因果、对比、拆解）
ANALYSIS_INDICATORS = [
    "因为", "所以", "原因是", "之所以", "导致",
    "对比", "相比", "区别在于", "差异", "不同",
    "拆解", "分解", "分为", "几个层面", "几个维度",
    "为什么", "凭什么", "怎么会",
]
# 综合层指标词（框架、本质、规律）
SYNTHESIS_INDICATORS = [
    "本质", "核心", "关键", "底层", "根本",
    "框架", "体系", "模型", "范式", "结构",
    "规律", "趋势", "周期", "必然", "逻辑链",
    "第一性", "底层逻辑",
]
# 元认知层指标词（自我反思、方法论）
META_INDICATORS = [
    "回头看", "其实我之前", "我之前说的", "反思",
    "换个角度", "重新理解", "重新审视",
    "方法论", "思维模型", "认知", "认知偏差",
    "我自己也在", "我也在调整", "说实话我",
]

# ── 认知负荷指标 ──
# 口癖词（用于负荷估算）
FILLER_WORDS = ["呃", "嗯", "那个", "这个", "啊", "就是", "然后", "其实", "可能", " basically"]


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def _time_segment_label(index: int, total: int, duration: float) -> str:
    """生成时间分段标签。"""
    seg_dur = duration / total if total > 0 else 0
    start_min = index * seg_dur / 60
    end_min = (index + 1) * seg_dur / 60
    return f"{start_min:.0f}-{end_min:.0f}min"


def _compute_trend(values: List[float]) -> str:
    """判断指标趋势: improving / declining / stable。"""
    if len(values) < 2:
        return "stable"
    mid = len(values) // 2
    first_half = values[:mid] if mid > 0 else values[:1]
    second_half = values[mid:] if mid > 0 else values[1:]
    avg_first = sum(first_half) / len(first_half) if first_half else 0
    avg_second = sum(second_half) / len(second_half) if second_half else 0
    diff = avg_second - avg_first
    if abs(diff) < 0.05 or (avg_first > 0 and abs(diff) / avg_first < 0.05):
        return "stable"
    return "improving" if diff > 0 else "declining"


def _count_keywords(text: str, keywords: List[str]) -> int:
    """统计文本中关键词出现总次数。"""
    return sum(text.count(kw) for kw in keywords)


def _get_ngrams(text: str, n: int = 2) -> set:
    """提取字符级 n-gram 集合。"""
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _split_sentences(text: str) -> List[str]:
    """按标点分割句子。"""
    parts = re.split(r"[。！？；\n]", text)
    return [p.strip() for p in parts if p.strip()]


# ══════════════════════════════════════════════════════════════
# 1. 论证链追踪
# ══════════════════════════════════════════════════════════════
def _extract_argument_structure_rule(
    response_text: str,
) -> Dict[str, Any]:
    """基于规则从回应文本中提取论证结构。

    识别论证三要素：主张(claim)、论据(evidence)、结论(conclusion)。
    通过连接词标记和句式模式做规则匹配。

    Args:
        response_text: 回应文本。

    Returns:
        论证结构字典:
            has_claim: 是否包含主张
            has_evidence: 是否包含论据
            has_conclusion: 是否包含结论
            claim_sentences: 主张句子列表
            evidence_sentences: 论据句子列表
            conclusion_sentences: 结论句子列表
            logic_connectors: 使用的逻辑连接词统计
            completeness: 论证完整性 [0-1]
    """
    sentences = _split_sentences(response_text)

    claims: List[str] = []
    evidences: List[str] = []
    conclusions: List[str] = []
    connectors: Dict[str, int] = {}

    for sent in sentences:
        # 主张检测
        if any(sent.startswith(m) or m in sent[:10] for m in CLAIM_MARKERS):
            claims.append(sent)
        # 论据检测
        if any(m in sent for m in EVIDENCE_MARKERS):
            evidences.append(sent)
        # 结论检测
        if any(m in sent for m in CONCLUSION_MARKERS):
            conclusions.append(sent)

        # 连接词统计
        for marker in CAUSAL_MARKERS:
            if marker in sent:
                connectors["causal"] = connectors.get("causal", 0) + 1
        for marker in PROGRESSIVE_MARKERS:
            if marker in sent:
                connectors["progressive"] = connectors.get("progressive", 0) + 1
        for marker in SUMMARY_MARKERS:
            if marker in sent:
                connectors["summary"] = connectors.get("summary", 0) + 1
        for marker in EXAMPLE_MARKERS:
            if marker in sent:
                connectors["example"] = connectors.get("example", 0) + 1
        for marker in TRANSITION_MARKERS:
            if marker in sent:
                connectors["transition"] = connectors.get("transition", 0) + 1

    # 完整性：三要素齐全 = 1.0，两个 = 0.67，一个 = 0.33，零个 = 0
    elements = sum([bool(claims), bool(evidences), bool(conclusions)])
    completeness = safe_float(elements / 3.0)

    return {
        "has_claim": bool(claims),
        "has_evidence": bool(evidences),
        "has_conclusion": bool(conclusions),
        "claim_count": len(claims),
        "evidence_count": len(evidences),
        "conclusion_count": len(conclusions),
        "logic_connectors": connectors,
        "completeness": completeness,
    }


def _detect_chain_continuation(
    current_response: str,
    previous_responses: List[str],
) -> float:
    """检测当前回应是否延续之前的论证链。

    通过 n-gram 重叠度判断当前回应是否引用了之前的论点。

    Args:
        current_response: 当前回应文本。
        previous_responses: 之前所有回应文本列表。

    Returns:
        延续度 [0-1]，1 表示高度延续。
    """
    if not previous_responses or len(current_response) < MIN_CHARS_FOR_ANALYSIS:
        return 0.0

    current_ngrams = _get_ngrams(current_response)
    # 取最近3个回应做对比
    recent_prev = previous_responses[-3:] if len(previous_responses) >= 3 else previous_responses
    prev_ngrams: set = set()
    for r in recent_prev:
        prev_ngrams.update(_get_ngrams(r))

    if not current_ngrams or not prev_ngrams:
        return 0.0

    overlap = len(current_ngrams & prev_ngrams) / len(current_ngrams)
    return safe_float(min(1.0, overlap))


def _detect_logic_jumps(
    structure: Dict[str, Any],
    response_text: str,
) -> bool:
    """检测逻辑跳跃：有结论但缺乏论据支撑。

    Args:
        structure: 论证结构分析结果。
        response_text: 回应文本。

    Returns:
        是否存在逻辑跳跃。
    """
    # 有结论但无论据 → 逻辑跳跃
    if structure["has_conclusion"] and not structure["has_evidence"]:
        return True
    # 结论句很短（少于15字）且无因果连接词
    if structure["has_conclusion"]:
        conclusion_short = len(response_text) < 30
        has_causal = any(m in response_text for m in CAUSAL_MARKERS)
        if conclusion_short and not has_causal:
            return True
    return False


def track_argument_chains(
    exchanges: List[Dict[str, Any]],
    target_speaker: int,
    duration: float,
    llm_backend: Any = None,
    config: Optional[Dict[str, Any]] = None,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """论证链追踪分析。

    对每个交换对的主播回应，提取论证结构（主张→论据→结论），
    追踪论证链的延续性，检测逻辑跳跃。

    Args:
        exchanges: 交换对列表。
        target_speaker: 目标说话人 ID。
        duration: 会话总时长（秒）。
        llm_backend: 可选的 LLM 后端。
        config: 可选的分析配置。
        num_segments: 时间分段数量。

    Returns:
        论证链分析结果:
            avg_completeness: 平均论证完整性 [0-1]
            complete_arguments: 完整论证数量（三要素齐全）
            logic_jump_count: 逻辑跳跃次数
            avg_chain_continuation: 平均论证链延续度
            time_segments: 各时间段的论证质量
            trend: 趋势
            details: 各交换对的论证结构详情
    """
    if not exchanges:
        return {
            "avg_completeness": 0.0,
            "complete_arguments": 0,
            "logic_jump_count": 0,
            "avg_chain_continuation": 0.0,
            "time_segments": [],
            "trend": "stable",
            "details": [],
        }

    # 尝试初始化 LLM 内容分析器
    content_analyzer = None
    use_llm = False
    if llm_backend is not None:
        try:
            from analyzer.llm_analyzer import ContentAnalyzer
            content_analyzer = ContentAnalyzer(llm_backend=llm_backend, config=config)
            use_llm = not content_analyzer.is_mock
        except Exception:
            use_llm = False

    sorted_exchanges = sorted(exchanges, key=lambda x: x.get("question_time", 0))
    previous_responses: List[str] = []
    details: List[Dict[str, Any]] = []
    completeness_values: List[float] = []
    continuation_values: List[float] = []
    logic_jump_count = 0
    complete_count = 0

    for ex in sorted_exchanges:
        resp = ex.get("response", {})
        resp_text = resp.get("text", "").strip() if isinstance(resp, dict) else ""

        if len(resp_text) < MIN_CHARS_FOR_ANALYSIS:
            details.append({
                "time": ex.get("question_time_str", ""),
                "completeness": 0.0,
                "has_logic_jump": False,
                "chain_continuation": 0.0,
                "level": "skipped",
            })
            completeness_values.append(0.0)
            continuation_values.append(0.0)
            previous_responses.append(resp_text)
            continue

        # 提取论证结构
        if use_llm and content_analyzer is not None:
            structure = content_analyzer.extract_argument_structure(resp_text)
        else:
            structure = _extract_argument_structure_rule(resp_text)

        # 论证链延续
        continuation = _detect_chain_continuation(resp_text, previous_responses)

        # 逻辑跳跃
        has_jump = _detect_logic_jumps(structure, resp_text)
        if has_jump:
            logic_jump_count += 1

        # 完整论证
        if structure["completeness"] >= 1.0:
            complete_count += 1

        details.append({
            "time": ex.get("question_time_str", ""),
            "completeness": structure["completeness"],
            "has_claim": structure["has_claim"],
            "has_evidence": structure["has_evidence"],
            "has_conclusion": structure["has_conclusion"],
            "logic_connectors": structure.get("logic_connectors", {}),
            "has_logic_jump": has_jump,
            "chain_continuation": continuation,
            "response_preview": resp_text[:60],
        })

        completeness_values.append(structure["completeness"])
        continuation_values.append(continuation)
        previous_responses.append(resp_text)

    # 整体统计
    avg_completeness = safe_float(
        sum(completeness_values) / len(completeness_values) if completeness_values else 0
    )
    avg_continuation = safe_float(
        sum(continuation_values) / len(continuation_values) if continuation_values else 0
    )

    # 时间分段
    seg_dur = duration / num_segments if duration > 0 else 0
    segments: List[Dict[str, Any]] = []
    seg_completeness: List[float] = []

    for i in range(num_segments):
        seg_start = i * seg_dur
        seg_end = (i + 1) * seg_dur
        seg_details = [
            d for d, ex in zip(details, sorted_exchanges)
            if seg_start <= ex.get("question_time", 0) < seg_end
        ]
        if seg_details:
            seg_comp = safe_float(
                sum(d["completeness"] for d in seg_details) / len(seg_details)
            )
            seg_jumps = sum(1 for d in seg_details if d["has_logic_jump"])
        else:
            seg_comp = 0.0
            seg_jumps = 0
        seg_completeness.append(seg_comp)
        segments.append({
            "segment": _time_segment_label(i, num_segments, duration),
            "avg_completeness": seg_comp,
            "logic_jumps": seg_jumps,
            "exchanges": len(seg_details),
        })

    trend = _compute_trend(seg_completeness)

    return {
        "avg_completeness": avg_completeness,
        "complete_arguments": complete_count,
        "complete_ratio": safe_float(complete_count / len(sorted_exchanges)),
        "logic_jump_count": logic_jump_count,
        "logic_jump_ratio": safe_float(logic_jump_count / len(sorted_exchanges)),
        "avg_chain_continuation": avg_continuation,
        "time_segments": segments,
        "trend": trend,
        "details": details[:30],
        "analysis_mode": "llm" if use_llm else "rule",
    }


# ══════════════════════════════════════════════════════════════
# 2. 思维深度评估
# ══════════════════════════════════════════════════════════════
def _classify_depth_rule(response_text: str) -> Tuple[str, float]:
    """基于规则分类思维深度。

    通过指标词频率和逻辑连接词复杂度判断思维深度层级。

    Args:
        response_text: 回应文本。

    Returns:
        元组 (depth_level, confidence):
            depth_level: surface/analysis/synthesis/meta
            confidence: 分类置信度 [0-1]
    """
    if len(response_text) < MIN_CHARS_FOR_ANALYSIS:
        return DEPTH_SURFACE, 0.60

    # 统计各层指标词
    surface_score = _count_keywords(response_text, SURFACE_INDICATORS)
    analysis_score = _count_keywords(response_text, ANALYSIS_INDICATORS)
    synthesis_score = _count_keywords(response_text, SYNTHESIS_INDICATORS)
    meta_score = _count_keywords(response_text, META_INDICATORS)

    # 加权评分（高层级指标权重更高）
    scores = {
        DEPTH_SURFACE: surface_score * 1.0,
        DEPTH_ANALYSIS: analysis_score * 1.5,
        DEPTH_SYNTHESIS: synthesis_score * 2.0,
        DEPTH_META: meta_score * 2.5,
    }

    # 如果没有任何指标词命中，根据文本特征做默认判断
    total_score = sum(scores.values())
    if total_score == 0:
        # 短回应 → surface
        if len(response_text) < 30:
            return DEPTH_SURFACE, 0.55
        # 有因果词 → analysis
        if any(m in response_text for m in CAUSAL_MARKERS):
            return DEPTH_ANALYSIS, 0.55
        # 默认 surface
        return DEPTH_SURFACE, 0.40

    # 取最高分层级
    best_level = max(scores, key=scores.get)
    best_score = scores[best_level]
    confidence = safe_float(min(0.85, best_score / max(total_score, 1) + 0.3))

    # 如果高层级和低层级分数接近，取高层级（宁高勿低）
    if scores[DEPTH_META] > 0 and scores[DEPTH_META] >= scores[DEPTH_SURFACE] * 0.5:
        return DEPTH_META, max(confidence, 0.55)
    if scores[DEPTH_SYNTHESIS] > 0 and scores[DEPTH_SYNTHESIS] >= scores[DEPTH_SURFACE] * 0.5:
        return DEPTH_SYNTHESIS, max(confidence, 0.55)

    return best_level, confidence


def assess_thinking_depth(
    exchanges: List[Dict[str, Any]],
    target_speaker: int,
    duration: float,
    llm_backend: Any = None,
    config: Optional[Dict[str, Any]] = None,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """思维深度评估。

    对每个交换对的主播回应，分类思维深度层级（表面/分析/综合/元认知），
    统计深度分布和趋势。

    Args:
        exchanges: 交换对列表。
        target_speaker: 目标说话人 ID。
        duration: 会话总时长（秒）。
        llm_backend: 可选的 LLM 后端。
        config: 可选的分析配置。
        num_segments: 时间分段数量。

    Returns:
        思维深度分析结果:
            depth_distribution: 各层级分布
            avg_depth_score: 平均深度评分 [1-4]
            surface_dominance: 表面层占比
            high_level_ratio: 高层级（综合+元认知）占比
            time_segments: 各时间段的深度
            trend: 趋势
            details: 各交换对的深度分类详情
    """
    if not exchanges:
        return {
            "depth_distribution": {},
            "avg_depth_score": 0.0,
            "surface_dominance": 0.0,
            "high_level_ratio": 0.0,
            "time_segments": [],
            "trend": "stable",
            "details": [],
        }

    # 尝试初始化 LLM
    content_analyzer = None
    use_llm = False
    if llm_backend is not None:
        try:
            from analyzer.llm_analyzer import ContentAnalyzer
            content_analyzer = ContentAnalyzer(llm_backend=llm_backend, config=config)
            use_llm = not content_analyzer.is_mock
        except Exception:
            use_llm = False

    sorted_exchanges = sorted(exchanges, key=lambda x: x.get("question_time", 0))
    details: List[Dict[str, Any]] = []
    depth_scores: List[float] = []
    depth_levels: List[str] = []

    for ex in sorted_exchanges:
        resp = ex.get("response", {})
        resp_text = resp.get("text", "").strip() if isinstance(resp, dict) else ""

        if len(resp_text) < MIN_CHARS_FOR_ANALYSIS:
            details.append({
                "time": ex.get("question_time_str", ""),
                "depth_level": DEPTH_SURFACE,
                "confidence": 0.50,
                "depth_score": 1.0,
                "level": "skipped",
            })
            depth_scores.append(1.0)
            depth_levels.append(DEPTH_SURFACE)
            continue

        # 分类思维深度
        if use_llm and content_analyzer is not None:
            depth_level, confidence = content_analyzer.classify_thinking_depth(resp_text)
        else:
            depth_level, confidence = _classify_depth_rule(resp_text)

        score = float(DEPTH_WEIGHTS.get(depth_level, 1))

        details.append({
            "time": ex.get("question_time_str", ""),
            "depth_level": depth_level,
            "confidence": confidence,
            "depth_score": score,
            "response_preview": resp_text[:60],
        })
        depth_scores.append(score)
        depth_levels.append(depth_level)

    # 统计
    level_counts = Counter(depth_levels)
    total = len(depth_levels)
    distribution = {
        level: {
            "count": level_counts.get(level, 0),
            "ratio": safe_float(level_counts.get(level, 0) / total),
        }
        for level in ALL_DEPTH_LEVELS
    }

    avg_score = safe_float(sum(depth_scores) / len(depth_scores) if depth_scores else 0)
    surface_dominance = safe_float(level_counts.get(DEPTH_SURFACE, 0) / total)
    high_level_count = level_counts.get(DEPTH_SYNTHESIS, 0) + level_counts.get(DEPTH_META, 0)
    high_level_ratio = safe_float(high_level_count / total)

    # 时间分段
    seg_dur = duration / num_segments if duration > 0 else 0
    segments: List[Dict[str, Any]] = []
    seg_depths: List[float] = []

    for i in range(num_segments):
        seg_start = i * seg_dur
        seg_end = (i + 1) * seg_dur
        seg_details = [
            (d, ex) for d, ex in zip(details, sorted_exchanges)
            if seg_start <= ex.get("question_time", 0) < seg_end
        ]
        if seg_details:
            seg_avg = safe_float(
                sum(d["depth_score"] for d, _ in seg_details) / len(seg_details)
            )
            seg_surface = sum(1 for d, _ in seg_details if d["depth_level"] == DEPTH_SURFACE)
        else:
            seg_avg = 0.0
            seg_surface = 0
        seg_depths.append(seg_avg)
        segments.append({
            "segment": _time_segment_label(i, num_segments, duration),
            "avg_depth_score": seg_avg,
            "surface_count": seg_surface,
            "exchanges": len(seg_details),
        })

    trend = _compute_trend(seg_depths)

    return {
        "depth_distribution": distribution,
        "avg_depth_score": avg_score,
        "surface_dominance": surface_dominance,
        "high_level_ratio": high_level_ratio,
        "time_segments": segments,
        "trend": trend,
        "details": details[:30],
        "analysis_mode": "llm" if use_llm else "rule",
    }


# ══════════════════════════════════════════════════════════════
# 3. 认知负荷与内容质量关联
# ══════════════════════════════════════════════════════════════
def _compute_cognitive_load_per_segment(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    duration: float,
    num_segments: int,
) -> List[Dict[str, Any]]:
    """按时间段计算认知负荷指标。

    对目标说话人的句子按时间分段，计算每段的口癖密度、
    词组重复、结巴模式，估算认知负荷等级。

    Args:
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        duration: 会话总时长。
        num_segments: 分段数量。

    Returns:
        各时间段的负荷指标列表。
    """
    target_sentences = [s for s in sentences if s.get("speaker") == target_speaker]
    if not target_sentences or duration <= 0:
        return []

    seg_dur = duration / num_segments
    segments: List[Dict[str, Any]] = []

    for i in range(num_segments):
        seg_start = i * seg_dur
        seg_end = (i + 1) * seg_dur
        seg_sents = [
            s for s in target_sentences
            if seg_start <= s.get("start", 0) < seg_end
        ]

        if not seg_sents:
            segments.append({
                "segment": _time_segment_label(i, num_segments, duration),
                "filler_count": 0,
                "filler_density": 0.0,
                "word_repetition": 0,
                "stutter": 0,
                "load_level": "unknown",
                "load_score": 0.0,
                "sentence_count": 0,
            })
            continue

        seg_text = "".join(s.get("text", "") for s in seg_sents)
        seg_chars = len(seg_text)

        # 口癖密度
        filler_count = _count_keywords(seg_text, FILLER_WORDS)
        filler_density = safe_float(filler_count / max(seg_chars, 1) * 1000)

        # 词组重复
        word_reps = len(re.findall(r"(.{2,3})\1{1,}", seg_text))

        # 结巴
        stutters = len(re.findall(r"(.)\1{2,}", seg_text))

        # 负荷等级
        if word_reps > 15 or filler_count > 120:
            load_level = "critical"
            load_score = 4.0
        elif word_reps > 8 or filler_count > 80:
            load_level = "high"
            load_score = 3.0
        elif word_reps > 4 or filler_count > 50:
            load_level = "medium"
            load_score = 2.0
        else:
            load_level = "low"
            load_score = 1.0

        segments.append({
            "segment": _time_segment_label(i, num_segments, duration),
            "filler_count": filler_count,
            "filler_density": filler_density,
            "word_repetition": word_reps,
            "stutter": stutters,
            "load_level": load_level,
            "load_score": load_score,
            "sentence_count": len(seg_sents),
        })

    return segments


def _extract_content_quality_per_segment(
    content_result: Dict[str, Any],
    duration: float,
    num_segments: int,
) -> List[Dict[str, Any]]:
    """从内容层分析结果中提取各时间段的内容质量指标。

    Args:
        content_result: content_metrics 的分析结果。
        duration: 会话总时长。
        num_segments: 分段数量。

    Returns:
        各时间段的内容质量指标列表。
    """
    segments: List[Dict[str, Any]] = []

    # 接球率时间段
    catch_segments = content_result.get("catch_rate_analysis", {}).get("time_segments", [])
    # 信息增量时间段
    increment_segments = content_result.get("information_density", {}).get("time_segments", [])
    # 话题连贯性时间段
    coherence_result = content_result.get("topic_coherence", {})

    for i in range(num_segments):
        catch_rate = catch_segments[i]["catch_rate"] if i < len(catch_segments) else 0
        increment = increment_segments[i].get("avg_increment", 0) if i < len(increment_segments) else 0
        seg_label = _time_segment_label(i, num_segments, duration)

        segments.append({
            "segment": seg_label,
            "catch_rate": safe_float(catch_rate),
            "information_increment": safe_float(increment),
        })

    return segments


def _classify_quality_load_zone(
    load_level: str,
    catch_rate: float,
    increment: float,
) -> str:
    """根据负荷等级和内容质量，分类时间段的认知-内容状态。

    四种状态:
    - resilient: 高负荷下仍保持高质量（压力下能接球）
    - overload: 高负荷且质量下降（认知过载）
    - coasting: 低负荷且质量也低（懈怠）
    - optimal: 低负荷且质量高（最佳状态）

    Args:
        load_level: 负荷等级 (low/medium/high/critical)。
        catch_rate: 接球率。
        increment: 信息增量密度。

    Returns:
        状态标签。
    """
    high_load = load_level in ("high", "critical")
    # 内容质量综合判断：接球率和增量的加权
    quality = catch_rate * 0.5 + increment * 0.5
    high_quality = quality > 0.35

    if high_load and high_quality:
        return "resilient"
    elif high_load and not high_quality:
        return "overload"
    elif not high_load and high_quality:
        return "optimal"
    else:
        return "coasting"


def correlate_cognitive_content(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    content_result: Dict[str, Any],
    duration: float,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """认知负荷与内容质量关联分析。

    将认知负荷时间分段与内容质量时间分段对齐，
    分析负荷变化对内容质量的影响。

    Args:
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        content_result: content_metrics 的分析结果。
        duration: 会话总时长（秒）。
        num_segments: 时间分段数量。

    Returns:
        关联分析结果:
            load_segments: 各时间段负荷指标
            quality_segments: 各时间段内容质量
            correlation: 负荷与质量的关联分析
            overload_zones: 认知过载区列表
            optimal_zones: 最佳状态区列表
            quality_degradation: 质量退化分析
    """
    if duration <= 0:
        return {
            "load_segments": [],
            "quality_segments": [],
            "correlation": {},
            "overload_zones": [],
            "optimal_zones": [],
            "quality_degradation": {},
        }

    # 计算各时间段负荷
    load_segments = _compute_cognitive_load_per_segment(
        sentences, target_speaker, duration, num_segments
    )

    # 提取各时间段内容质量
    quality_segments = _extract_content_quality_per_segment(
        content_result, duration, num_segments
    )

    # 对齐分析
    zones: List[Dict[str, Any]] = []
    load_scores: List[float] = []
    quality_scores: List[float] = []

    for ls, qs in zip(load_segments, quality_segments):
        load_level = ls.get("load_level", "unknown")
        catch_rate = qs.get("catch_rate", 0)
        increment = qs.get("information_increment", 0)

        zone_type = _classify_quality_load_zone(load_level, catch_rate, increment)

        zone = {
            "segment": ls.get("segment", ""),
            "load_level": load_level,
            "load_score": ls.get("load_score", 0),
            "catch_rate": catch_rate,
            "information_increment": increment,
            "filler_density": ls.get("filler_density", 0),
            "word_repetition": ls.get("word_repetition", 0),
            "zone_type": zone_type,
        }
        zones.append(zone)

        load_scores.append(ls.get("load_score", 0))
        # 内容质量综合分
        quality = catch_rate * 0.5 + increment * 0.5
        quality_scores.append(safe_float(quality))

    # 过载区和最佳区
    overload_zones = [z for z in zones if z["zone_type"] == "overload"]
    optimal_zones = [z for z in zones if z["zone_type"] == "optimal"]
    resilient_zones = [z for z in zones if z["zone_type"] == "resilient"]
    coasting_zones = [z for z in zones if z["zone_type"] == "coasting"]

    # 负荷-质量相关性（简单 Pearson）
    correlation = _pearson_correlation(load_scores, quality_scores)

    # 质量退化分析：高负荷段 vs 低负荷段的质量差
    high_load_quality = [
        q for q, l in zip(quality_scores, load_scores) if l >= 3.0
    ]
    low_load_quality = [
        q for q, l in zip(quality_scores, load_scores) if l <= 2.0
    ]

    has_low_baseline = bool(low_load_quality)

    if high_load_quality and low_load_quality:
        # 正常情况：有高有低，直接对比
        avg_high = sum(high_load_quality) / len(high_load_quality)
        avg_low = sum(low_load_quality) / len(low_load_quality)
        degradation = safe_float(avg_high - avg_low)
        degradation_pct = safe_float(
            degradation / avg_low * 100 if avg_low > 0 else 0
        )
    elif high_load_quality and not low_load_quality:
        # 全程高负荷：用最低负荷段做相对基线，检测时间维度退化
        paired = sorted(zip(load_scores, quality_scores))
        baseline_quality = paired[0][1] if paired else 0
        avg_high = sum(high_load_quality) / len(high_load_quality)
        degradation = safe_float(baseline_quality - avg_high)
        degradation_pct = safe_float(
            degradation / baseline_quality * 100
            if baseline_quality > 0 else 0
        )
    else:
        avg_high = 0.0
        degradation = 0.0
        degradation_pct = 0.0

    # 时间维度退化：前半段 vs 后半段质量差
    if len(quality_scores) >= 4:
        mid = len(quality_scores) // 2
        early_avg = sum(quality_scores[:mid]) / mid
        late_avg = sum(quality_scores[mid:]) / (len(quality_scores) - mid)
        temporal_degradation = safe_float(early_avg - late_avg)
    else:
        temporal_degradation = 0.0

    return {
        "zones": zones,
        "overload_zones": overload_zones,
        "optimal_zones": optimal_zones,
        "resilient_zones": resilient_zones,
        "coasting_zones": coasting_zones,
        "correlation": {
            "pearson_r": correlation,
            "interpretation": _interpret_correlation(correlation),
        },
        "quality_degradation": {
            "high_load_avg_quality": safe_float(
                sum(high_load_quality) / len(high_load_quality)
            ) if high_load_quality else 0,
            "low_load_avg_quality": safe_float(
                sum(low_load_quality) / len(low_load_quality)
            ) if low_load_quality else 0,
            "degradation": degradation,
            "degradation_pct": degradation_pct,
            "has_low_baseline": has_low_baseline,
            "temporal_degradation": temporal_degradation,
        },
        "zone_distribution": dict(Counter(z["zone_type"] for z in zones)),
    }


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """计算两个列表的 Pearson 相关系数。

    Args:
        x: 第一个数值列表。
        y: 第二个数值列表。

    Returns:
        相关系数 [-1, 1]。
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    x = x[:n]
    y = y[:n]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return safe_float(numerator / (denom_x * denom_y))


def _interpret_correlation(r: float) -> str:
    """解释相关系数。

    Args:
        r: Pearson 相关系数。

    Returns:
        解释文字。
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        direction = "无"
    elif abs_r < 0.3:
        direction = "弱"
    elif abs_r < 0.5:
        direction = "中等"
    elif abs_r < 0.7:
        direction = "较强"
    else:
        direction = "强"

    if r > 0:
        return f"{direction}正相关（负荷升高，质量也升高）"
    elif r < 0:
        return f"{direction}负相关（负荷升高，质量下降）"
    else:
        return "无相关性"


# ══════════════════════════════════════════════════════════════
# 4. 综合评分
# ══════════════════════════════════════════════════════════════
def compute_cognitive_score(
    argument_result: Dict[str, Any],
    depth_result: Dict[str, Any],
    correlation_result: Dict[str, Any],
) -> Dict[str, float]:
    """计算认知层综合评分。

    评分维度和权重:
      - 论证完整性 (35%): 完整论证比例 + 完整性均值
      - 思维深度 (35%): 深度评分 + 高层级占比
      - 认知韧性 (30%): 过载区比例的逆指标 + 质量退化控制

    Args:
        argument_result: 论证链分析结果。
        depth_result: 思维深度分析结果。
        correlation_result: 认知-内容关联结果。

    Returns:
        评分字典:
            argument_score: 论证完整性得分 [0-100]
            thinking_depth_score: 思维深度得分 [0-100]
            cognitive_resilience_score: 认知韧性得分 [0-100]
            cognitive_quality_score: 认知质量综合得分 [0-100]
    """
    # 论证完整性得分
    avg_completeness = argument_result.get("avg_completeness", 0)
    complete_ratio = argument_result.get("complete_ratio", 0)
    jump_ratio = argument_result.get("logic_jump_ratio", 0)
    argument_score = safe_float(
        avg_completeness * 50 + complete_ratio * 30 + (1 - jump_ratio) * 20
    )

    # 思维深度得分
    avg_depth = depth_result.get("avg_depth_score", 0)
    high_level_ratio = depth_result.get("high_level_ratio", 0)
    surface_dominance = depth_result.get("surface_dominance", 0)
    # 深度评分映射：1→0, 2→33, 3→67, 4→100
    depth_score = safe_float((avg_depth - 1) / 3 * 100)
    depth_score = safe_float(
        depth_score * 0.5 + high_level_ratio * 100 * 0.3 + (1 - surface_dominance) * 100 * 0.2
    )

    # 认知韧性得分
    zones = correlation_result.get("zones", [])
    total_zones = len(zones) if zones else 1
    overload_count = len(correlation_result.get("overload_zones", []))
    optimal_count = len(correlation_result.get("optimal_zones", []))
    resilient_count = len(correlation_result.get("resilient_zones", []))
    coasting_count = len(correlation_result.get("coasting_zones", []))
    deg_info = correlation_result.get("quality_degradation", {})
    degradation_pct = abs(deg_info.get("degradation_pct", 0))
    temporal_deg = abs(deg_info.get("temporal_degradation", 0))
    has_low_baseline = deg_info.get("has_low_baseline", True)

    # 负荷水平分布
    load_scores_list = [z.get("load_score", 0) for z in zones]
    high_load_count = sum(1 for s in load_scores_list if s >= 3.0)
    low_load_count = sum(1 for s in load_scores_list if s <= 2.0)
    sustained_high_ratio = safe_float(
        high_load_count / total_zones if total_zones else 0
    )

    # 韧性 = 过载区少 + 最佳区多 + 韧性区适度加分 + 退化控制 + 低负荷比例
    overload_penalty = safe_float(overload_count / total_zones * 100)
    # 最佳区（低负荷+高质量）给满额加分
    optimal_bonus = safe_float(optimal_count / total_zones * 100)
    # 韧性区（高负荷但仍维持质量）给半额加分
    resilient_bonus = safe_float(resilient_count / total_zones * 50)
    # 退化惩罚：综合负荷退化 + 时间退化
    degradation_penalty = safe_float(
        min(100, degradation_pct * 0.6 + temporal_deg * 100 * 0.4)
    )
    # 全程高负荷惩罚：没有低负荷段时，说明持续承压
    sustained_penalty = safe_float(
        (1 - low_load_count / total_zones) * 15 if total_zones else 0
    ) if not has_low_baseline else 0

    resilience_score = safe_float(
        max(0, min(100,
            100
            - overload_penalty * 0.4
            - degradation_penalty * 0.3
            - sustained_penalty
            + optimal_bonus * 0.3
            + resilient_bonus * 0.2
        ))
    )

    # 综合
    weights = {
        "argument": 0.35,
        "depth": 0.35,
        "resilience": 0.30,
    }
    overall = (
        min(100, argument_score) * weights["argument"]
        + min(100, depth_score) * weights["depth"]
        + min(100, resilience_score) * weights["resilience"]
    )

    return {
        "argument_score": safe_float(min(100, argument_score)),
        "thinking_depth_score": safe_float(min(100, depth_score)),
        "cognitive_resilience_score": safe_float(min(100, resilience_score)),
        "cognitive_quality_score": safe_float(overall),
    }


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
def compute_cognitive_metrics(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    dialogue_result: Optional[Dict[str, Any]] = None,
    content_result: Optional[Dict[str, Any]] = None,
    duration: float = 0,
    llm_backend: Any = None,
    config: Optional[Dict[str, Any]] = None,
    num_segments: int = TIME_SEGMENTS,
) -> Dict[str, Any]:
    """认知层分析主入口。

    基于对话结构和内容层分析结果，计算认知层面的三大指标:
    1. 论证链追踪
    2. 思维深度评估
    3. 认知负荷与内容质量关联

    如果未提供 dialogue_result，会自动调用 parse_dialogue 生成。
    如果未提供 content_result，关联分析将仅使用负荷数据。

    Args:
        sentences: ASR 句子列表。
        target_speaker: 目标说话人 ID。
        dialogue_result: dialogue_parser 的输出。为 None 则自动生成。
        content_result: content_metrics 的输出。为 None 则关联分析降级。
        duration: 会话总时长（秒）。为 0 则从 sentences 推断。
        llm_backend: 可选的 LLM 后端实例。
        config: 可选的分析配置。
        num_segments: 时间分段数量。

    Returns:
        认知层分析结果字典，包含:
            argument_chains: 论证链追踪
            thinking_depth: 思维深度评估
            cognitive_content_correlation: 认知-内容关联
            summary: 综合评分
    """
    print("[cognitive_metrics] 认知层指标分析...")

    # 自动生成 dialogue_result
    if dialogue_result is None:
        from analyzer.dialogue_parser import parse_dialogue
        dialogue_result = parse_dialogue(
            sentences, target_speaker, llm_backend=llm_backend
        )

    exchanges = dialogue_result.get("exchanges", [])

    # 推断时长
    if duration <= 0 and sentences:
        duration = max(s["end"] for s in sentences)

    if not exchanges:
        print("  无交换对，跳过认知层分析")
        return {
            "argument_chains": {},
            "thinking_depth": {},
            "cognitive_content_correlation": {},
            "summary": {
                "cognitive_quality_score": 0.0,
                "note": "无交换对（独白场或单人发言）",
            },
        }

    # 1. 论证链追踪
    print("  [1/3] 论证链追踪...")
    argument_result = track_argument_chains(
        exchanges, target_speaker, duration, llm_backend, config, num_segments
    )
    print(
        f"    完整论证: {argument_result['complete_arguments']}/{argument_result.get('complete_ratio', 0):.0%}, "
        f"逻辑跳跃: {argument_result['logic_jump_count']}次, "
        f"趋势: {argument_result['trend']}"
    )

    # 2. 思维深度评估
    print("  [2/3] 思维深度评估...")
    depth_result = assess_thinking_depth(
        exchanges, target_speaker, duration, llm_backend, config, num_segments
    )
    print(
        f"    平均深度: {depth_result['avg_depth_score']:.2f}/4, "
        f"高层级占比: {depth_result['high_level_ratio']:.0%}, "
        f"趋势: {depth_result['trend']}"
    )

    # 3. 认知-内容关联
    print("  [3/3] 认知负荷与内容质量关联...")
    if content_result is None:
        content_result = {}
    correlation_result = correlate_cognitive_content(
        sentences, target_speaker, content_result, duration, num_segments
    )
    zone_dist = correlation_result.get("zone_distribution", {})
    print(
        f"    过载区: {len(correlation_result.get('overload_zones', []))}, "
        f"最佳区: {len(correlation_result.get('optimal_zones', []))}, "
        f"相关性: {correlation_result['correlation']['interpretation']}"
    )

    # 4. 综合评分
    scores = compute_cognitive_score(
        argument_result, depth_result, correlation_result
    )
    print(
        f"  认知质量综合得分: {scores['cognitive_quality_score']:.1f}/100"
    )

    return {
        "argument_chains": argument_result,
        "thinking_depth": depth_result,
        "cognitive_content_correlation": correlation_result,
        "summary": scores,
    }
