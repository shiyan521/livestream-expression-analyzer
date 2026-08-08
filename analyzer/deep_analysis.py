#!/usr/bin/env python3
"""
深度分析模块 - 对直播ASR结果进行表达结构、内容质量、收尾困难、认知负荷、即兴表达原理分析。

五个分析维度:
  1. 表达结构分析 (Expression Structure)
  2. 内容质量分析 (Content Quality)
  3. 收尾困难分析 (Ending Difficulty)
  4. 认知负荷分析 (Cognitive Load)
  5. 即兴表达认知模型评估 (Impromptu Expression Model)

用法:
  python -m analyzer.deep_analysis --asr-json <asr.json> --speaker 0 --output result.json

输出: JSON格式的深度分析结果，可直接用于报告生成。
"""
import argparse
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

# ═══════════════════════════════════════════════════════════════════════════════
# 默认词库常量（可通过函数参数覆盖）
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_FILLER_LIST: List[str] = [
    '呃', '嗯', '那个', '这个', '啊', '呢', '吧', '嘛', '然后', '就是', '可能', '其实',
    '就是说', '或者说', '这样一个', '样一个', '来去', '到底', '对对', '这个这个',
    '就是说', '比如说',
]

DEFAULT_LOGIC_CAUSAL: List[str] = ['因为', '所以', '因此', '导致', '由于']
DEFAULT_LOGIC_TRANSITION: List[str] = ['但是', '不过', '然而', '虽然', '尽管', '反而']
DEFAULT_LOGIC_PROGRESSIVE: List[str] = ['而且', '另外', '同时', '并且', '不仅', '甚至', '更重要的是']
DEFAULT_LOGIC_SUMMARY: List[str] = ['总之', '总的来说', '最终', '总结一下', '核心就是', '关键就是', '本质', '核心']
DEFAULT_LOGIC_SEQUENCE: List[str] = ['首先', '然后', '接着', '其次', '最后', '第一', '第二', '第三']
DEFAULT_LOGIC_EXAMPLE: List[str] = ['比如', '例如', '举个例子', '比如说', '像是']

DEFAULT_CONCRETE_WORDS: List[str] = [
    '看到', '听到', '发现', '观察', '展位', '人', '公司', '产品', '视频', '直播',
    '手机', '电脑', '图片', '门口', '展馆', '展台', '观众', '机器人', '屏幕', '展品',
]
DEFAULT_ABSTRACT_WORDS: List[str] = [
    '本质', '核心', '逻辑', '概念', '理论', '体系', '框架', '思维', '认知',
    '模式', '维度', '战略', '趋势', '未来', '价值', '意义', '哲学',
]

DEFAULT_TECH_TERMS: List[str] = [
    'AI', 'OPC', 'API', 'Agent', 'agent', '大模型', 'GPT', '模型', '云计算', 'SaaS',
    'LLM', '开源', '芯片', '算法', '数据', '平台', '工具', '应用', '技术', '开发',
    '编程', '代码', '部署', 'prompt', 'Prompt', 'token', '智能', '自动化',
]

DEFAULT_PERSONAL_WORDS: List[str] = [
    '我', '我的', '我觉得', '我认为', '我感觉', '我发现', '我看到', '我遇到',
    '我之前', '我们', '我跟',
]

# 保留旧名称供外部引用兼容
FILLER_LIST = DEFAULT_FILLER_LIST
LOGIC_CAUSAL = DEFAULT_LOGIC_CAUSAL
LOGIC_TRANSITION = DEFAULT_LOGIC_TRANSITION
LOGIC_PROGRESSIVE = DEFAULT_LOGIC_PROGRESSIVE
LOGIC_SUMMARY = DEFAULT_LOGIC_SUMMARY
LOGIC_SEQUENCE = DEFAULT_LOGIC_SEQUENCE
LOGIC_EXAMPLE = DEFAULT_LOGIC_EXAMPLE
CONCRETE_WORDS = DEFAULT_CONCRETE_WORDS
ABSTRACT_WORDS = DEFAULT_ABSTRACT_WORDS
TECH_TERMS = DEFAULT_TECH_TERMS
PERSONAL_WORDS = DEFAULT_PERSONAL_WORDS


# ═══════════════════════════════════════════════════════════════════════════════
# 词库配置容器
# ═══════════════════════════════════════════════════════════════════════════════
class LexiconConfig:
    """词库配置容器，允许用户自定义所有词库常量。

    所有字段均有默认值，用户只需覆盖需要修改的部分。

    Attributes:
        filler_list: 口癖词列表
        logic_causal: 因果逻辑词
        logic_transition: 转折逻辑词
        logic_progressive: 递进逻辑词
        logic_summary: 总结逻辑词
        logic_sequence: 顺序逻辑词
        logic_example: 举例逻辑词
        concrete_words: 具象词
        abstract_words: 抽象词
        tech_terms: 技术术语
        personal_words: 个人视角词
    """

    def __init__(
        self,
        filler_list: Optional[List[str]] = None,
        logic_causal: Optional[List[str]] = None,
        logic_transition: Optional[List[str]] = None,
        logic_progressive: Optional[List[str]] = None,
        logic_summary: Optional[List[str]] = None,
        logic_sequence: Optional[List[str]] = None,
        logic_example: Optional[List[str]] = None,
        concrete_words: Optional[List[str]] = None,
        abstract_words: Optional[List[str]] = None,
        tech_terms: Optional[List[str]] = None,
        personal_words: Optional[List[str]] = None,
    ) -> None:
        self.filler_list: List[str] = filler_list if filler_list is not None else DEFAULT_FILLER_LIST
        self.logic_causal: List[str] = logic_causal if logic_causal is not None else DEFAULT_LOGIC_CAUSAL
        self.logic_transition: List[str] = logic_transition if logic_transition is not None else DEFAULT_LOGIC_TRANSITION
        self.logic_progressive: List[str] = logic_progressive if logic_progressive is not None else DEFAULT_LOGIC_PROGRESSIVE
        self.logic_summary: List[str] = logic_summary if logic_summary is not None else DEFAULT_LOGIC_SUMMARY
        self.logic_sequence: List[str] = logic_sequence if logic_sequence is not None else DEFAULT_LOGIC_SEQUENCE
        self.logic_example: List[str] = logic_example if logic_example is not None else DEFAULT_LOGIC_EXAMPLE
        self.concrete_words: List[str] = concrete_words if concrete_words is not None else DEFAULT_CONCRETE_WORDS
        self.abstract_words: List[str] = abstract_words if abstract_words is not None else DEFAULT_ABSTRACT_WORDS
        self.tech_terms: List[str] = tech_terms if tech_terms is not None else DEFAULT_TECH_TERMS
        self.personal_words: List[str] = personal_words if personal_words is not None else DEFAULT_PERSONAL_WORDS


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_time(sec: float) -> str:
    """将秒数格式化为 MM:SS 字符串。

    Args:
        sec: 秒数

    Returns:
        格式化后的时间字符串，如 "05:30"
    """
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def count_keywords(text: str, keywords: Sequence[str]) -> int:
    """统计文本中关键词出现的总次数。

    Args:
        text: 待分析的文本
        keywords: 关键词列表

    Returns:
        所有关键词出现次数的总和
    """
    return sum(text.count(kw) for kw in keywords)


def count_filler(text: str, filler_list: Optional[List[str]] = None) -> int:
    """统计文本中口癖词出现的总次数。

    Args:
        text: 待分析的文本
        filler_list: 口癖词列表，默认使用 DEFAULT_FILLER_LIST

    Returns:
        所有口癖词出现次数的总和
    """
    fl = filler_list if filler_list is not None else DEFAULT_FILLER_LIST
    return sum(text.count(f) for f in fl)


def load_asr(json_path: str, target_speaker: int = 0) -> List[Dict[str, Any]]:
    """加载ASR JSON，返回目标说话人的句子列表。

    Args:
        json_path: ASR diarization JSON 文件路径
        target_speaker: 目标说话人ID

    Returns:
        句子列表，每个元素包含 start(秒), end(秒), text 三个字段，
        按时间顺序排列
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences: List[Dict[str, Any]] = []
    for t in data.get('transcripts', []):
        for s in t.get('sentences', []):
            if s['speaker_id'] == target_speaker:
                sentences.append({
                    'start': s['begin_time'] / 1000.0,
                    'end': s['end_time'] / 1000.0,
                    'text': s['text'],
                })
    sentences.sort(key=lambda x: x['start'])
    return sentences


def identify_turns(sentences: List[Dict[str, Any]], gap_threshold: float = 3.0) -> List[List[Dict[str, Any]]]:
    """识别轮次（间隔超过阈值秒数则视为新轮次）。

    Args:
        sentences: 句子列表（已按时间排序）
        gap_threshold: 轮次间隔阈值（秒），默认 3.0

    Returns:
        轮次列表，每个轮次是一个句子列表
    """
    if not sentences:
        return []
    turns: List[List[Dict[str, Any]]] = []
    current_turn: List[Dict[str, Any]] = [sentences[0]]
    for i in range(1, len(sentences)):
        gap = sentences[i]['start'] - sentences[i - 1]['end']
        if gap > gap_threshold:
            turns.append(current_turn)
            current_turn = [sentences[i]]
        else:
            current_turn.append(sentences[i])
    if current_turn:
        turns.append(current_turn)
    return turns


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 表达结构分析
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_expression_structure(
    sentences: List[Dict[str, Any]],
    config: Optional[LexiconConfig] = None,
) -> Dict[str, Any]:
    """表达结构分析。

    分析维度包括：句式类型分布、逻辑连接词密度、句长分布、
    自我修正频率、字符重复与词组重复模式。

    Args:
        sentences: 句子列表
        config: 词库配置，默认使用 DEFAULT 配置

    Returns:
        包含以下键的分析结果字典：
        - sentence_types: 句式类型统计（陈述/疑问/感叹）
        - logic_connectors: 逻辑连接词统计及密度
        - sentence_length: 句长分布
        - self_correction: 自我修正次数
        - char_repetition: 字符重复统计
        - word_repetition: 词组重复统计
    """
    if config is None:
        config = LexiconConfig()

    all_text = ''.join(s['text'] for s in sentences)
    total_chars = len(all_text)
    total_sents = len(sentences)

    # 句式类型
    question_count = sum(1 for s in sentences if '？' in s['text'] or '?' in s['text'])
    exclamation_count = sum(1 for s in sentences if '！' in s['text'] or '!' in s['text'])
    statement_count = total_sents - question_count - exclamation_count

    # 逻辑连接词
    logic_counts = {
        'causal': count_keywords(all_text, config.logic_causal),
        'transition': count_keywords(all_text, config.logic_transition),
        'progressive': count_keywords(all_text, config.logic_progressive),
        'summary': count_keywords(all_text, config.logic_summary),
        'sequence': count_keywords(all_text, config.logic_sequence),
        'example': count_keywords(all_text, config.logic_example),
    }
    logic: Dict[str, Any] = {}
    for k, v in logic_counts.items():
        logic[k] = v
        logic[f'{k}_density'] = round(v / max(total_chars, 1) * 1000, 1)

    # 句长分布
    lengths = [len(s['text']) for s in sentences]
    length_dist = {
        'short': sum(1 for l in lengths if l < 10),
        'medium': sum(1 for l in lengths if 10 <= l < 30),
        'long': sum(1 for l in lengths if 30 <= l < 60),
        'extra_long': sum(1 for l in lengths if l >= 60),
    }
    length_pct = {}
    for k, v in length_dist.items():
        length_pct[f'{k}_pct'] = round(v / max(total_sents, 1) * 100, 1)
    length_dist.update(length_pct)

    # 自我修正
    correction_patterns = ['我是说', '不对', '不是', '我的意思是', '说错了', '应该说',
                           '更准确地说', '换句话说']
    corrections = sum(all_text.count(p) for p in correction_patterns)

    # 字符重复（我我我）
    char_reps = re.findall(r'(.)\1{2,}', all_text)
    char_rep_counter = dict(Counter(char_reps).most_common(10))

    # 词组重复（就是就是）
    word_reps = re.findall(r'(.{2,3})\1{1,}', all_text)
    word_rep_counter = dict(Counter(word_reps).most_common(10))

    return {
        'sentence_types': {
            'statement': statement_count,
            'question': question_count,
            'exclamation': exclamation_count,
            'statement_pct': round(statement_count / max(total_sents, 1) * 100, 1),
            'question_pct': round(question_count / max(total_sents, 1) * 100, 1),
        },
        'logic_connectors': logic,
        'sentence_length': length_dist,
        'self_correction': corrections,
        'char_repetition': {
            'total': len(char_reps),
            'top_patterns': char_rep_counter,
        },
        'word_repetition': {
            'total': len(word_reps),
            'top_patterns': word_rep_counter,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 内容质量分析
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_content_quality(
    sentences: List[Dict[str, Any]],
    config: Optional[LexiconConfig] = None,
) -> Dict[str, Any]:
    """内容质量分析。

    分析维度包括：词汇丰富度（TTR）、具象词vs抽象词、技术术语密度、
    个人视角密度、信息密度（有效字数/口癖字数比）。

    Args:
        sentences: 句子列表
        config: 词库配置，默认使用 DEFAULT 配置

    Returns:
        包含以下键的分析结果字典：
        - vocabulary: 词汇丰富度
        - concrete_abstract: 具象/抽象词统计
        - tech_terms: 技术术语统计
        - personal_perspective: 个人视角统计
        - info_density: 信息密度统计
    """
    if config is None:
        config = LexiconConfig()

    all_text = ''.join(s['text'] for s in sentences)
    total_chars = len(all_text)

    # 词汇丰富度
    chars = [c for c in all_text if c.strip()]
    unique_chars = set(chars)
    ttr_char = round(len(unique_chars) / max(len(chars), 1), 3)

    words_2 = [all_text[i:i + 2] for i in range(len(all_text) - 1)]
    unique_w2 = set(words_2)
    ttr_w2 = round(len(unique_w2) / max(len(words_2), 1), 3)

    # 具体vs抽象
    concrete_count = count_keywords(all_text, config.concrete_words)
    abstract_count = count_keywords(all_text, config.abstract_words)
    concrete_abstract_ratio = round(concrete_count / max(abstract_count, 1), 1)

    # 技术术语
    tech_count = count_keywords(all_text, config.tech_terms)
    tech_density = round(tech_count / max(total_chars, 1) * 1000, 1)

    # 个人视角
    personal_count = count_keywords(all_text, config.personal_words)
    wo_de_count = all_text.count('我觉得')
    personal_density = round(personal_count / max(total_chars, 1) * 1000, 1)

    # 信息密度
    filler_chars = sum(all_text.count(f) * len(f) for f in config.filler_list)
    effective_chars = total_chars - filler_chars
    speaking_minutes = sum(s['end'] - s['start'] for s in sentences) / 60
    effective_density = round(effective_chars / max(speaking_minutes, 1))

    return {
        'vocabulary': {
            'ttr_char': ttr_char,
            'ttr_2gram': ttr_w2,
        },
        'concrete_abstract': {
            'concrete': concrete_count,
            'abstract': abstract_count,
            'ratio': concrete_abstract_ratio,
        },
        'tech_terms': {
            'count': tech_count,
            'density': tech_density,
        },
        'personal_perspective': {
            'total': personal_count,
            'wo_juede': wo_de_count,
            'density': personal_density,
        },
        'info_density': {
            'total_chars': total_chars,
            'filler_chars': filler_chars,
            'filler_pct': round(filler_chars / max(total_chars, 1) * 100, 1),
            'effective_chars': effective_chars,
            'effective_density': effective_density,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 收尾困难分析
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_ending_difficulty(
    sentences: List[Dict[str, Any]],
    turns: List[List[Dict[str, Any]]],
    config: Optional[LexiconConfig] = None,
) -> Dict[str, Any]:
    """收尾困难分析。

    分析维度包括：超长末句比例、末句绕圈检测、语法断裂检测、
    开场vs收尾逻辑词/口癖密度对比、"所以"使用模式分析、典型案例。

    Args:
        sentences: 句子列表
        turns: 轮次列表（由 identify_turns 生成）
        config: 词库配置，默认使用 DEFAULT 配置

    Returns:
        包含以下键的分析结果字典：
        - extra_long_last: 超长末句统计
        - circling: 末句绕圈检测
        - grammar_break: 语法断裂检测
        - opening_vs_ending: 开场vs收尾对比
        - suoyi_usage: "所以"使用模式
        - typical_cases: 典型案例列表
    """
    if config is None:
        config = LexiconConfig()

    long_turns = [t for t in turns if len(t) >= 3]
    if not long_turns:
        return {'error': 'no long turns'}

    last_sentences = [t[-1] for t in long_turns]
    middle_sentences = [s for t in long_turns for s in t[1:-1]]

    # 超长末句
    last_lens = [len(s['text']) for s in last_sentences]
    extra_long_last = sum(1 for l in last_lens if l > 60)
    extra_long_last_pct = round(extra_long_last / max(len(last_sentences), 1) * 100, 1)

    # 末句绕圈检测
    circle_count = 0
    for turn in long_turns:
        last = turn[-1]['text']
        if len(last) < 30:
            continue
        last_third = last[-len(last) // 3:]
        prev = ''.join(s['text'] for s in turn[:-1])
        last_words = set(last_third[j:j + 3] for j in range(len(last_third) - 2))
        overlap = sum(1 for w in last_words if w in prev)
        if overlap > 3:
            circle_count += 1
    circle_pct = round(circle_count / max(len(long_turns), 1) * 100, 1)

    # 语法断裂检测
    break_count = 0
    for turn in long_turns:
        last = turn[-1]['text']
        signals = 0
        if last.count('，') >= 4 and len(last) > 50:
            signals += 1
        if re.search(r'(就是|然后|其实|可能|这个).{0,5}(就是|然后|其实|可能|这个)', last):
            signals += 1
        if last.rstrip().endswith(('啊', '吧', '呢', '嘛')) and any(f in last[-15:] for f in ['呃', '嗯', '那个', '就是', '其实']):
            signals += 1
        if len(last) > 100:
            signals += 1
        if '所以' in last:
            after_suoyi = last[last.index('所以'):]
            if not any(w in after_suoyi for w in ['我觉得', '我认为', '核心', '关键', '本质']):
                signals += 1
        if signals >= 2:
            break_count += 1
    break_pct = round(break_count / max(len(long_turns), 1) * 100, 1)

    # 开场vs收尾
    logic_words = (config.logic_causal + config.logic_transition + config.logic_progressive
                   + config.logic_summary + config.logic_sequence)
    filler_simple = ['呃', '嗯', '那个', '这个', '啊', '就是', '然后', '其实']

    all_first_logic = all_last_logic = 0
    all_first_filler = all_last_filler = 0
    all_first_chars = all_last_chars = 0
    count = 0
    for turn in turns:
        if len(turn) < 6:
            continue
        count += 1
        first_3 = turn[:3]
        last_3 = turn[-3:]
        first_text = ''.join(s['text'] for s in first_3)
        last_text = ''.join(s['text'] for s in last_3)
        all_first_logic += count_keywords(first_text, logic_words)
        all_last_logic += count_keywords(last_text, logic_words)
        all_first_filler += count_keywords(first_text, filler_simple)
        all_last_filler += count_keywords(last_text, filler_simple)
        all_first_chars += len(first_text)
        all_last_chars += len(last_text)

    # "所以"分析
    all_text = ''.join(s['text'] for s in sentences)
    suoyi_contexts: List[str] = []
    for s in sentences:
        if '所以' in s['text']:
            idx = s['text'].index('所以')
            suoyi_contexts.append(s['text'][idx:])
    suoyi_after_lens = [len(c) for c in suoyi_contexts]

    # 典型案例
    cases: List[Dict[str, Any]] = []
    for i, turn in enumerate(long_turns):
        last = turn[-1]
        last_text = last['text']
        chaos = 0
        if len(last_text) > 60:
            chaos += 2
        if len(last_text) > 100:
            chaos += 2
        chaos += min(count_keywords(last_text, ['然后', '就是', '其实', '可能', '所以', '这个']), 5)
        if count_filler(last_text, config.filler_list) >= 3:
            chaos += 2
        if chaos >= 5:
            cases.append({
                'time': fmt_time(last['start']),
                'text': last_text[:120],
                'length': len(last_text),
                'chaos': chaos,
            })
    cases.sort(key=lambda x: -x['chaos'])

    return {
        'extra_long_last': {
            'count': extra_long_last,
            'total': len(last_sentences),
            'pct': extra_long_last_pct,
        },
        'circling': {
            'count': circle_count,
            'total': len(long_turns),
            'pct': circle_pct,
        },
        'grammar_break': {
            'count': break_count,
            'total': len(long_turns),
            'pct': break_pct,
        },
        'opening_vs_ending': {
            'analyzed_turns': count,
            'opening_logic_density': round(all_first_logic / max(all_first_chars, 1) * 1000, 1),
            'ending_logic_density': round(all_last_logic / max(all_last_chars, 1) * 1000, 1),
            'opening_filler_density': round(all_first_filler / max(all_first_chars, 1) * 1000, 1),
            'ending_filler_density': round(all_last_filler / max(all_last_chars, 1) * 1000, 1),
        },
        'suoyi_usage': {
            'total': len(suoyi_contexts),
            'avg_chars_after': round(sum(suoyi_after_lens) / max(len(suoyi_after_lens), 1)),
            'over_30': sum(1 for l in suoyi_after_lens if l > 30),
            'over_50': sum(1 for l in suoyi_after_lens if l > 50),
        },
        'typical_cases': cases[:8],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 认知负荷分析
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_cognitive_load(
    sentences: List[Dict[str, Any]],
    config: Optional[LexiconConfig] = None,
) -> Dict[str, Any]:
    """认知负荷分析。

    按每10分钟时间段分析结巴模式、词组重复、口癖密度，
    并评估各时段的认知负荷等级（low/medium/high/critical）。

    Args:
        sentences: 句子列表
        config: 词库配置，默认使用 DEFAULT 配置

    Returns:
        包含以下键的分析结果字典：
        - time_segments: 各时间段的负荷分析
        - stutter_patterns: 结巴模式统计
    """
    if config is None:
        config = LexiconConfig()

    all_text = ''.join(s['text'] for s in sentences)

    # 按时间段
    time_segments: List[Dict[str, Any]] = []
    for start_min in range(0, int(sentences[-1]['start'] // 60) + 10, 10):
        end_min = start_min + 10
        seg_sents = [s for s in sentences if start_min * 60 <= s['start'] < end_min * 60]
        if not seg_sents:
            continue
        seg_text = ''.join(s['text'] for s in seg_sents)
        stutter = len(re.findall(r'(.)\1{2,}', seg_text))
        word_rep = len(re.findall(r'(.{2,3})\1{1,}', seg_text))
        filler = count_keywords(seg_text, ['呃', '嗯', '那个', '这个', '就是', '然后', '其实'])
        load = 'low'
        if word_rep > 15 or filler > 120:
            load = 'critical'
        elif word_rep > 8 or filler > 80:
            load = 'high'
        elif word_rep > 4 or filler > 50:
            load = 'medium'
        time_segments.append({
            'period': f'{start_min}-{end_min}min',
            'sentences': len(seg_sents),
            'stutter': stutter,
            'word_repetition': word_rep,
            'filler': filler,
            'load': load,
        })

    # 结巴模式
    stutter_patterns = re.findall(r'(.)\1{2,}', all_text)
    stutter_counter = dict(Counter(stutter_patterns).most_common(10))

    return {
        'time_segments': time_segments,
        'stutter_patterns': {
            'total': len(stutter_patterns),
            'top': stutter_counter,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 即兴表达认知模型
# ═══════════════════════════════════════════════════════════════════════════════
def assess_impromptu_model(
    structure: Dict[str, Any],
    content: Dict[str, Any],
    ending: Dict[str, Any],
    cognitive: Dict[str, Any],
) -> Dict[str, Any]:
    """即兴表达认知模型评估。

    基于信息提取、信息组织、语言输出三个维度评估即兴表达能力，
    并判断主要瓶颈所在。

    Args:
        structure: 表达结构分析结果（analyze_expression_structure 返回值）
        content: 内容质量分析结果（analyze_content_quality 返回值）
        ending: 收尾困难分析结果（analyze_ending_difficulty 返回值）
        cognitive: 认知负荷分析结果（analyze_cognitive_load 返回值）

    Returns:
        包含以下键的评估结果字典：
        - extraction: 信息提取能力（strong/weak）
        - organization: 信息组织能力（strong/medium/weak）
        - output: 语言输出能力（strong/medium/weak）
        - bottleneck: 主要瓶颈（extraction/organization/output/none）
    """
    # 信息提取：看具体词和场景描述
    extraction = 'strong' if content['concrete_abstract']['concrete'] > 100 else 'weak'

    # 信息组织：看抽象词、收尾质量
    org_signals = 0
    if content['concrete_abstract']['ratio'] > 10:
        org_signals += 1
    if ending['circling']['pct'] > 30:
        org_signals += 1
    if ending['suoyi_usage']['avg_chars_after'] > 30:
        org_signals += 1
    if ending['grammar_break']['pct'] > 15:
        org_signals += 1
    organization = 'weak' if org_signals >= 3 else ('medium' if org_signals >= 1 else 'strong')

    # 语言输出：看口癖和结巴
    output_signals = 0
    if content['info_density']['filler_pct'] > 12:
        output_signals += 1
    if structure['char_repetition']['total'] > 20:
        output_signals += 1
    if structure['word_repetition']['total'] > 70:
        output_signals += 1
    output = 'weak' if output_signals >= 2 else ('medium' if output_signals >= 1 else 'strong')

    bottleneck = 'organization' if organization == 'weak' else \
        ('output' if output == 'weak' else
         ('extraction' if extraction == 'weak' else 'none'))

    return {
        'extraction': extraction,
        'organization': organization,
        'output': output,
        'bottleneck': bottleneck,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 完整分析入口
# ═══════════════════════════════════════════════════════════════════════════════
def run_deep_analysis(
    sentences: List[Dict[str, Any]],
    config: Optional[LexiconConfig] = None,
) -> Dict[str, Any]:
    """执行完整的深度分析流程。

    依次执行五个维度的分析，并汇总为完整结果。

    Args:
        sentences: 句子列表（由 load_asr 返回）
        config: 词库配置，默认使用 DEFAULT 配置

    Returns:
        完整的深度分析结果字典，包含 meta 和五个维度的分析结果
    """
    if config is None:
        config = LexiconConfig()

    turns = identify_turns(sentences)

    structure = analyze_expression_structure(sentences, config)
    content = analyze_content_quality(sentences, config)
    ending = analyze_ending_difficulty(sentences, turns, config)
    cognitive = analyze_cognitive_load(sentences, config)
    impromptu = assess_impromptu_model(structure, content, ending, cognitive)

    result = {
        'meta': {
            'total_sentences': len(sentences),
            'total_turns': len(turns),
            'total_chars': sum(len(s['text']) for s in sentences),
        },
        'expression_structure': structure,
        'content_quality': content,
        'ending_difficulty': ending,
        'cognitive_load': cognitive,
        'impromptu_model': impromptu,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """命令行入口：解析参数并执行深度分析。"""
    parser = argparse.ArgumentParser(description='深度表达分析模板')
    parser.add_argument('--asr-json', required=True, help='ASR diarization JSON 路径')
    parser.add_argument('--result-json', help='analyze.py 结果 JSON（仅作参考）')
    parser.add_argument('--speaker', type=int, default=0, help='目标说话人 ID')
    parser.add_argument('--output', default='deep_analysis_result.json', help='输出 JSON 路径')
    args = parser.parse_args()

    print(f"[Deep Analysis] Loading ASR from {args.asr_json}...")
    sentences = load_asr(args.asr_json, args.speaker)
    print(f"  Loaded {len(sentences)} sentences for speaker {args.speaker}")

    result = run_deep_analysis(sentences)
    result['meta']['speaker'] = args.speaker

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDone! Output saved to {args.output}")

    # 打印摘要
    structure = result['expression_structure']
    content = result['content_quality']
    ending = result['ending_difficulty']
    impromptu = result['impromptu_model']

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Extra-long sentences (>60字): {structure['sentence_length']['extra_long_pct']}%")
    print(f"Word group repetition: {structure['word_repetition']['total']}")
    print(f"Concrete:Abstract ratio: {content['concrete_abstract']['ratio']}:1")
    print(f"'我觉得' count: {content['personal_perspective']['wo_juede']}")
    print(f"Info density: {content['info_density']['effective_density']} chars/min")
    print(f"Last sentence circling: {ending['circling']['pct']}%")
    print(f"Last sentence grammar break: {ending['grammar_break']['pct']}%")
    print(f"'所以' avg chars after: {ending['suoyi_usage']['avg_chars_after']}")
    print(f"Cognitive bottleneck: {impromptu['bottleneck']}")


if __name__ == '__main__':
    main()
