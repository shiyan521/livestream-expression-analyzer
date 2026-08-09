"""
对话结构解析器模块。

从 ASR 句子列表构建对话结构，识别 Q&A 对，分类回应类型。
这是内容层分析的基础模块，为对话接球率和信息增量密度提供结构化输入。

核心概念:
  - Turn (轮次): 合并相邻同说话人片段后的连续发言段
  - Exchange (交换): 非目标说话人轮次 + 目标说话人轮次的配对
  - Response Type (回应类型): extend/shift/avoid/repeat 四分类

分类策略:
  - 规则优先: 通过句首词、响应长度、文本相似度做初步分类
  - LLM 增强: 规则置信度低时，调用 LLM 做语义分类（可选）

用法:
  from analyzer.dialogue_parser import parse_dialogue
  dialogue = parse_dialogue(sentences, target_speaker=0)
  print(dialogue["summary"]["catch_rate"])
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from analyzer.utils import fmt_time, safe_float


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 回应类型
RESPONSE_EXTEND = "extend"   # 延展：接住对方内容并添加新信息
RESPONSE_SHIFT = "shift"     # 转移：切换话题或未回应对方内容
RESPONSE_AVOID = "avoid"     # 回避：极短回应或敷衍
RESPONSE_REPEAT = "repeat"   # 重复：重复已有内容，无新增信息

ALL_RESPONSE_TYPES = [RESPONSE_EXTEND, RESPONSE_SHIFT, RESPONSE_AVOID, RESPONSE_REPEAT]

# 有效回应类型（接球率分子只算 extend）
EFFECTIVE_RESPONSE_TYPES = {RESPONSE_EXTEND}

# 句首赞同标记（通常表示接住了对方内容）
AGREEMENT_MARKERS = [
    "对", "是的", "没错", "对对", "嗯对", "确实",
    "同意", "有道理", "说得对", "是这个", "对，",
    "是的，", "没错，",
]

# 话题转移标记（可能表示未接住对方内容）
SHIFT_MARKERS = [
    "说到", "其实我想说", "说到这个", "不过我觉得",
    "但是我觉得", "我想说的是", "另外", "换个角度",
    "其实关键", "我觉得更重要的",
]

# 回避标记
AVOID_MARKERS = [
    "这个不好说", "怎么说呢", "看情况", "不好讲",
    "差不多吧", "就这样", "嗯嗯", "好吧",
]

# 最小有效回应长度（字符数）
MIN_RESPONSE_CHARS = 8

# 轮次合并间隔（秒）：相邻同说话人片段间隔小于此值则合并
TURN_MERGE_GAP = 5.0


# ══════════════════════════════════════════════════════════════
# 轮次构建
# ══════════════════════════════════════════════════════════════
def build_turns(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 ASR 句子列表合并为对话轮次。

    相邻同说话人的句子合并为一个轮次，间隔超过 TURN_MERGE_GAP 秒则断开。

    Args:
        sentences: ASR 句子列表，每个句子包含 start, end, text, speaker 字段。

    Returns:
        轮次列表，每个轮次包含:
            speaker: 说话人 ID
            start: 开始时间（秒）
            end: 结束时间（秒）
            text: 合并后的完整文本
            sentence_indices: 包含的原始句子索引列表
    """
    if not sentences:
        return []

    turns: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for i, s in enumerate(sentences):
        if current is None:
            current = {
                "speaker": s["speaker"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "sentence_indices": [i],
            }
        elif (
            s["speaker"] == current["speaker"]
            and s["start"] - current["end"] <= TURN_MERGE_GAP
        ):
            # 合并到当前轮次
            current["end"] = s["end"]
            current["text"] += s["text"]
            current["sentence_indices"].append(i)
        else:
            turns.append(current)
            current = {
                "speaker": s["speaker"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "sentence_indices": [i],
            }

    if current:
        turns.append(current)

    return turns


# ══════════════════════════════════════════════════════════════
# 交换对识别
# ══════════════════════════════════════════════════════════════
def identify_exchanges(
    turns: List[Dict[str, Any]],
    target_speaker: int,
) -> List[Dict[str, Any]]:
    """从轮次列表中识别 Q&A 交换对。

    一个交换对由「非目标说话人轮次」+「目标说话人轮次」组成。
    连续多个目标说话人轮次只取第一个作为回应。

    Args:
        turns: 轮次列表。
        target_speaker: 目标说话人 ID。

    Returns:
        交换对列表，每个交换对包含:
            question: 非目标说话人的轮次（提问/观点抛出）
            response: 目标说话人的轮次（回应）
            question_time: 提问时间戳（秒）
            response_time: 回应时间戳（秒）
    """
    exchanges: List[Dict[str, Any]] = []

    for i in range(len(turns) - 1):
        curr_turn = turns[i]
        next_turn = turns[i + 1]

        # 非目标说话人 → 目标说话人
        if (
            curr_turn["speaker"] != target_speaker
            and next_turn["speaker"] == target_speaker
        ):
            exchanges.append({
                "question": curr_turn,
                "response": next_turn,
                "question_time": curr_turn["start"],
                "response_time": next_turn["start"],
                "response_latency": safe_float(
                    next_turn["start"] - curr_turn["end"]
                ),
            })

    return exchanges


# ══════════════════════════════════════════════════════════════
# 回应类型分类（规则）
# ══════════════════════════════════════════════════════════════
def _text_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的 Jaccard 相似度（基于字符级 n-gram）。

    Args:
        text_a: 文本 A。
        text_b: 文本 B。

    Returns:
        相似度 [0, 1]，1 表示完全相同。
    """
    if not text_a or not text_b:
        return 0.0

    # 字符级 2-gram
    def get_ngrams(text: str, n: int = 2) -> set:
        text = text.replace(" ", "").replace("\n", "")
        if len(text) < n:
            return {text}
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    ngrams_a = get_ngrams(text_a)
    ngrams_b = get_ngrams(text_b)

    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union) if union else 0.0


def _check_start_markers(text: str, markers: List[str]) -> bool:
    """检查文本开头是否包含给定标记词。

    Args:
        text: 待检查文本。
        markers: 标记词列表。

    Returns:
        是否以任一标记词开头。
    """
    text_stripped = text.strip()
    for marker in markers:
        if text_stripped.startswith(marker):
            return True
    return False


def classify_response_rule_based(
    question: Dict[str, Any],
    response: Dict[str, Any],
    previous_responses: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """基于规则的回应类型分类。

    分类逻辑（按优先级）:
    1. 回应极短（< MIN_RESPONSE_CHARS）→ avoid
    2. 回应开头有回避标记 → avoid
    3. 回应与之前某次回应高度相似（> 0.6）→ repeat
    4. 回应开头有转移标记 → shift
    5. 回应开头有赞同标记 + 回应足够长 → extend
    6. 回应与问题有语义关联（相似度 > 0.15）→ extend
    7. 默认 → shift

    Args:
        question: 提问轮次。
        response: 回应轮次。
        previous_responses: 之前所有回应的文本列表（用于重复检测）。

    Returns:
        元组 (response_type, confidence):
            response_type: extend/shift/avoid/repeat
            confidence: 分类置信度 [0, 1]
    """
    resp_text = response["text"].strip()
    resp_len = len(resp_text)
    ques_text = question["text"].strip()

    # 1. 极短回应 → avoid
    if resp_len < MIN_RESPONSE_CHARS:
        return RESPONSE_AVOID, 0.85

    # 2. 回避标记
    if _check_start_markers(resp_text, AVOID_MARKERS) and resp_len < 30:
        return RESPONSE_AVOID, 0.75

    # 3. 重复检测：与之前回应高度相似
    if previous_responses:
        max_sim = 0.0
        for prev_resp in previous_responses:
            sim = _text_similarity(resp_text, prev_resp)
            if sim > max_sim:
                max_sim = sim
        if max_sim > 0.6:
            return RESPONSE_REPEAT, 0.70

    # 4. 转移标记
    if _check_start_markers(resp_text, SHIFT_MARKERS):
        # 但如果后面又回到了问题主题，可能是 extend
        # 这里简化处理：有转移标记但回应很长且与问题有关联 → extend
        sim_with_question = _text_similarity(resp_text, ques_text)
        if sim_with_question > 0.2 and resp_len > 50:
            return RESPONSE_EXTEND, 0.55
        return RESPONSE_SHIFT, 0.65

    # 5. 赞同标记 + 足够长 → extend
    has_agreement = _check_start_markers(resp_text, AGREEMENT_MARKERS)
    if has_agreement and resp_len >= 20:
        return RESPONSE_EXTEND, 0.75

    # 6. 与问题有语义关联 → extend
    sim_with_question = _text_similarity(resp_text, ques_text)
    if sim_with_question > 0.15:
        return RESPONSE_EXTEND, 0.60

    # 7. 回应很长但没有明显关联 → shift
    if resp_len > 100:
        return RESPONSE_SHIFT, 0.50

    # 8. 默认 → shift
    return RESPONSE_SHIFT, 0.40


# ══════════════════════════════════════════════════════════════
# LLM 增强分类
# ══════════════════════════════════════════════════════════════
def classify_response_with_llm(
    question: Dict[str, Any],
    response: Dict[str, Any],
    llm_backend: Any,
    prompt_config: Optional[Dict[str, str]] = None,
) -> Tuple[str, float]:
    """使用 LLM 做回应类型分类。

    当规则分类置信度低时，调用此方法做语义级分类。

    Args:
        question: 提问轮次。
        response: 回应轮次。
        llm_backend: LLM 后端实例（需实现 chat 方法）。
        prompt_config: prompt 配置，包含 system_prompt 和 user_template。
            若为 None 则使用默认 prompt。

    Returns:
        元组 (response_type, confidence)。
    """
    ques_text = question["text"].strip()[:500]
    resp_text = response["text"].strip()[:500]

    if not ques_text or not resp_text:
        return RESPONSE_SHIFT, 0.30

    # 默认 prompt
    if prompt_config is None:
        prompt_config = {
            "system_prompt": (
                "你是一个对话分析专家。你的任务是分析对话中回应者对提问的回应类型。\n"
                "回应类型有四种：\n"
                "- extend: 延展——接住对方内容并添加新信息或观点\n"
                "- shift: 转移——切换话题或未回应对方的内容\n"
                "- avoid: 回避——极短回应或敷衍\n"
                "- repeat: 重复——重复已有内容，无新增信息\n"
                "请只输出一个单词：extend、shift、avoid 或 repeat。"
            ),
            "user_template": (
                "嘉宾说：{question}\n\n"
                "主播回应：{response}\n\n"
                "请分类这个回应的类型。"
            ),
        }

    user_prompt = prompt_config["user_template"].format(
        question=ques_text,
        response=resp_text,
    )

    try:
        result = llm_backend.chat(
            system_prompt=prompt_config["system_prompt"],
            user_prompt=user_prompt,
            temperature=0.1,
        )
        result = result.strip().lower()

        # 解析 LLM 输出
        for rtype in ALL_RESPONSE_TYPES:
            if rtype in result:
                return rtype, 0.80

        # 无法解析
        return RESPONSE_SHIFT, 0.30
    except Exception as e:
        print(f"  [dialogue_parser] LLM分类失败: {e}")
        return RESPONSE_SHIFT, 0.20


# ══════════════════════════════════════════════════════════════
# 核心观点提取（规则版）
# ══════════════════════════════════════════════════════════════
def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """从文本中提取关键短语（规则版）。

    基于标点符号分割 + 长度过滤 + 去重。

    Args:
        text: 输入文本。
        max_phrases: 最大短语数量。

    Returns:
        关键短语列表。
    """
    # 按标点分割
    segments = re.split(r"[。！？，；、\n]", text)
    # 过滤：长度 5-50，去重
    seen = set()
    phrases: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if 5 <= len(seg) <= 50 and seg not in seen:
            seen.add(seg)
            phrases.append(seg)
        if len(phrases) >= max_phrases:
            break
    return phrases


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
def parse_dialogue(
    sentences: List[Dict[str, Any]],
    target_speaker: int,
    llm_backend: Any = None,
    prompt_config: Optional[Dict[str, str]] = None,
    llm_confidence_threshold: float = 0.55,
) -> Dict[str, Any]:
    """解析对话结构，识别 Q&A 对并分类回应类型。

    Args:
        sentences: ASR 句子列表，每个句子包含 start, end, text, speaker 字段。
        target_speaker: 目标说话人 ID。
        llm_backend: 可选的 LLM 后端实例。当规则分类置信度低于
            llm_confidence_threshold 时，使用 LLM 做增强分类。
            为 None 则纯规则分类。
        prompt_config: LLM 分类用的 prompt 配置。
        llm_confidence_threshold: 规则分类置信度低于此值时触发 LLM 分类。

    Returns:
        对话分析结果字典，包含:
            turns: 所有轮次列表
            exchanges: 所有交换对列表（含分类结果）
            summary: 汇总统计
                - total_exchanges: 交换对总数
                - catch_rate: 接球率（extend 比例）
                - response_type_distribution: 各类型分布
                - avg_response_latency: 平均回应延迟
    """
    print("[dialogue_parser] 解析对话结构...")

    # 1. 构建轮次
    turns = build_turns(sentences)
    print(f"  轮次数: {len(turns)}")

    # 2. 识别交换对
    exchanges = identify_exchanges(turns, target_speaker)
    print(f"  交换对数: {len(exchanges)}")

    if not exchanges:
        print("  无交换对（可能是独白场或单人发言）")
        return {
            "turns": turns,
            "exchanges": [],
            "summary": {
                "total_exchanges": 0,
                "catch_rate": 0.0,
                "response_type_distribution": {},
                "avg_response_latency": 0.0,
            },
        }

    # 3. 分类回应类型
    previous_responses: List[str] = []
    llm_used_count = 0

    for ex in exchanges:
        # 规则分类
        rtype, confidence = classify_response_rule_based(
            ex["question"], ex["response"], previous_responses
        )

        # 置信度低 → LLM 增强
        if confidence < llm_confidence_threshold and llm_backend is not None:
            llm_type, llm_conf = classify_response_with_llm(
                ex["question"], ex["response"], llm_backend, prompt_config
            )
            if llm_conf > confidence:
                rtype = llm_type
                confidence = llm_conf
                llm_used_count += 1

        ex["response_type"] = rtype
        ex["confidence"] = safe_float(confidence)
        ex["question_preview"] = ex["question"]["text"][:60]
        ex["response_preview"] = ex["response"]["text"][:60]
        ex["question_time_str"] = fmt_time(ex["question_time"])
        ex["response_time_str"] = fmt_time(ex["response_time"])

        previous_responses.append(ex["response"]["text"])

    # 4. 汇总统计
    type_counts = Counter(ex["response_type"] for ex in exchanges)
    total = len(exchanges)
    catch_rate = safe_float(type_counts.get(RESPONSE_EXTEND, 0) / total) if total > 0 else 0
    avg_latency = safe_float(
        sum(ex["response_latency"] for ex in exchanges) / total
    ) if total > 0 else 0

    distribution = {rtype: type_counts.get(rtype, 0) for rtype in ALL_RESPONSE_TYPES}

    print(
        f"  接球率: {catch_rate:.1%}, "
        f"延展={distribution[RESPONSE_EXTEND]}, "
        f"转移={distribution[RESPONSE_SHIFT]}, "
        f"回避={distribution[RESPONSE_AVOID]}, "
        f"重复={distribution[RESPONSE_REPEAT]}"
    )
    if llm_used_count > 0:
        print(f"  LLM增强分类: {llm_used_count}/{total} 次")

    return {
        "turns": turns,
        "exchanges": exchanges,
        "summary": {
            "total_exchanges": total,
            "catch_rate": catch_rate,
            "response_type_distribution": distribution,
            "response_type_percent": {
                rtype: safe_float(type_counts.get(rtype, 0) / total * 100)
                if total > 0 else 0
                for rtype in ALL_RESPONSE_TYPES
            },
            "avg_response_latency": avg_latency,
            "llm_enhanced_count": llm_used_count,
        },
    }
