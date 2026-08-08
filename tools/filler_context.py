#!/usr/bin/env python3
"""
口癖上下文分析工具。

对ASR逐字稿进行口癖密度、口癖组合、句式模式、时间段聚合、
高口癖上下文等多维度分析。

用法:
  python -m tools.filler_context --asr-json <asr.json> --speaker 0

输出: 控制台打印分析结果。
"""
import argparse
import json
from typing import Any, Dict, List, Optional, Sequence


# ══════════════════════════════════════════════════════════════
# 默认口癖词库
# ══════════════════════════════════════════════════════════════
DEFAULT_FILLER_CORE: List[str] = [
    "来去", "呃", "那个", "或者说", "样一个", "比如说", "这样一个",
    "到底", "嘛", "就是说", "呢", "嗯", "这个这个", "吧", "对对",
]
DEFAULT_FILLER_EXTENDED: List[str] = [
    "这个", "啊", "那", "就是", "然后", "可能", "所以", "的话",
    "其实", "还是", "你可以", "比较", "确实", "OK", "你觉得",
]


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def ms_to_time(ms: int) -> str:
    """毫秒转 mm:ss 格式字符串。

    Args:
        ms: 毫秒数

    Returns:
        格式化后的时间字符串，如 "05:30"
    """
    s = ms // 1000
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"


def count_fillers(text: str, all_fillers: Sequence[str]) -> List[str]:
    """统计句子中的口癖数量和具体词。

    Args:
        text: 待分析的文本
        all_fillers: 口癖词列表

    Returns:
        找到的口癖词列表（含重复）
    """
    text_lower = text.lower()
    found: List[str] = []
    for word in all_fillers:
        count = text_lower.count(word.lower())
        found.extend([word] * count)
    return found


def calc_density(text: str, fillers: Sequence[str]) -> float:
    """计算口癖密度（口癖字数/总字数）。

    Args:
        text: 待分析的文本
        fillers: 找到的口癖词列表

    Returns:
        口癖密度（0~1之间的浮点数）
    """
    if not text:
        return 0.0
    filler_chars = sum(len(f) for f in fillers)
    return filler_chars / len(text)


def load_asr_sentences(
    json_path: str,
    target_speaker: int = 0,
) -> List[Dict[str, Any]]:
    """加载ASR JSON，返回目标说话人的句子列表。

    Args:
        json_path: ASR diarization JSON 文件路径
        target_speaker: 目标说话人ID

    Returns:
        句子列表（原始 ASR sentence 格式）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences: List[Dict[str, Any]] = []
    for t in data.get("transcripts", []):
        for s in t.get("sentences", []):
            if s.get("speaker_id") == target_speaker:
                sentences.append(s)
    return sentences


# ══════════════════════════════════════════════════════════════
# 分析函数
# ══════════════════════════════════════════════════════════════
def analyze_filler_context(
    sentences: List[Dict[str, Any]],
    filler_core: Optional[List[str]] = None,
    filler_extended: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """分析每个句子的口癖上下文。

    Args:
        sentences: ASR 句子列表
        filler_core: 核心口癖词列表，默认使用 DEFAULT_FILLER_CORE
        filler_extended: 扩展口癖词列表，默认使用 DEFAULT_FILLER_EXTENDED

    Returns:
        分析结果列表，每个元素包含 index, time, text, fillers,
        core_count, ext_count, density, char_count 字段
    """
    if filler_core is None:
        filler_core = DEFAULT_FILLER_CORE
    if filler_extended is None:
        filler_extended = DEFAULT_FILLER_EXTENDED

    all_fillers = filler_core + filler_extended

    results: List[Dict[str, Any]] = []
    for i, s in enumerate(sentences):
        text = s.get("text", "")
        fillers = count_fillers(text, all_fillers)
        density = calc_density(text, fillers)
        core_fillers = [f for f in fillers if f in filler_core]

        results.append({
            "index": i,
            "time": ms_to_time(s.get("begin_time", 0)),
            "text": text,
            "fillers": fillers,
            "core_count": len(core_fillers),
            "ext_count": len(fillers) - len(core_fillers),
            "density": density,
            "char_count": len(text),
        })

    return results


def print_high_filler_sentences(
    results: List[Dict[str, Any]],
    density_threshold: float = 0.20,
    min_chars: int = 15,
    top_n: int = 20,
) -> None:
    """打印口癖密度最高的句子。

    Args:
        results: 分析结果列表
        density_threshold: 密度阈值，默认 0.20
        min_chars: 最小字符数过滤，默认 15
        top_n: 打印前 N 条，默认 20
    """
    high_filler = [r for r in results if r["density"] > density_threshold and r["char_count"] > min_chars]
    high_filler.sort(key=lambda x: x["density"], reverse=True)

    print(f"\n=== 口癖密度最高的句子（密度>{density_threshold:.0%}，字数>{min_chars}，"
          f"共{len(high_filler)}句）TOP{top_n} ===\n")
    for r in high_filler[:top_n]:
        print(f"[{r['time']}] 密度:{r['density']:.1%} | 核心:{r['core_count']} | 扩展:{r['ext_count']}")
        print(f"    {r['text']}")
        print(f"    口癖: {r['fillers']}")
        print()


def print_combo_sentences(
    results: List[Dict[str, Any]],
    min_fillers: int = 3,
    min_chars: int = 10,
    top_n: int = 20,
) -> None:
    """打印口癖Combo句子（一句话里多个口癖）。

    Args:
        results: 分析结果列表
        min_fillers: 最小口癖数量阈值，默认 3
        min_chars: 最小字符数过滤，默认 10
        top_n: 打印前 N 条，默认 20
    """
    combo = [r for r in results if len(r["fillers"]) >= min_fillers and r["char_count"] > min_chars]
    combo.sort(key=lambda x: len(x["fillers"]), reverse=True)

    print(f"\n=== 口癖Combo句子（一句话>={min_fillers}个口癖，共{len(combo)}句）TOP{top_n} ===\n")
    for r in combo[:top_n]:
        print(f"[{r['time']}] {len(r['fillers'])}个口癖")
        print(f"    {r['text']}")
        print(f"    口癖: {r['fillers']}")
        print()


def print_pattern_analysis(results: List[Dict[str, Any]]) -> None:
    """打印句式模式分析。

    分析以"那"开头的句子、出现多次"这个"/"就是"的句子。

    Args:
        results: 分析结果列表
    """
    print("\n=== 句式模式分析 ===\n")

    # 找"那"开头的句子
    na_sentences = [r for r in results if r["text"].startswith("那")]
    na_with_filler = [r for r in na_sentences if len(r["fillers"]) >= 1]
    if na_sentences:
        print(f"以'那'开头的句子: {len(na_sentences)}句，"
              f"其中带口癖: {len(na_with_filler)}句 "
              f"({len(na_with_filler)/len(na_sentences)*100:.0f}%)")

    # 找"这个"密集的区域
    zhege_sentences = [r for r in results if r["text"].count("这个") >= 2]
    print(f"\n一句话出现>=2次'这个': {len(zhege_sentences)}句")
    for r in zhege_sentences[:10]:
        print(f"  [{r['time']}] {r['text']}")

    # 找"就是"密集的区域
    jiushi_sentences = [r for r in results if r["text"].count("就是") >= 2]
    print(f"\n一句话出现>=2次'就是': {len(jiushi_sentences)}句")
    for r in jiushi_sentences[:10]:
        print(f"  [{r['time']}] {r['text']}")


def print_time_segment_density(results: List[Dict[str, Any]]) -> None:
    """按时间段聚合口癖密度（每10分钟一段）。

    Args:
        results: 分析结果列表
    """
    print("\n=== 时间段口癖密度 ===\n")
    segments: Dict[str, Dict[str, int]] = {}
    for r in results:
        time_str = r["time"]
        try:
            parts = time_str.split(":")
            mins = int(parts[0])
            seg = mins // 10
            key = f"{seg*10:02d}:00-{(seg+1)*10:02d}:00"
            if key not in segments:
                segments[key] = {"total_chars": 0, "total_filler_chars": 0, "sentences": 0}
            segments[key]["total_chars"] += r["char_count"]
            segments[key]["total_filler_chars"] += sum(len(f) for f in r["fillers"])
            segments[key]["sentences"] += 1
        except (ValueError, IndexError):
            pass

    for key in sorted(segments.keys()):
        seg = segments[key]
        if seg["total_chars"] > 0:
            density = seg["total_filler_chars"] / seg["total_chars"]
            print(f"{key}: 口癖密度 {density:.1%} | {seg['sentences']}句 | {seg['total_chars']}字")


def print_high_filler_context(
    results: List[Dict[str, Any]],
    density_threshold: float = 0.25,
    min_chars: int = 15,
    top_n: int = 15,
) -> None:
    """打印高口癖句子的上下文（前后2句）。

    Args:
        results: 分析结果列表
        density_threshold: 密度阈值，默认 0.25
        min_chars: 最小字符数过滤，默认 15
        top_n: 打印前 N 条，默认 15
    """
    print(f"\n=== 高口癖句子的上下文（密度>{density_threshold:.0%}）===\n")
    very_high = [r for r in results if r["density"] > density_threshold and r["char_count"] > min_chars]
    for r in very_high[:top_n]:
        idx = r["index"]
        print(f"--- [{r['time']}] 密度:{r['density']:.1%} ---")
        # 前2句
        for j in range(max(0, idx - 2), idx):
            print(f"  前文: {results[j]['text']}")
        print(f"  >>当前: {r['text']}")
        print(f"  >>口癖: {r['fillers']}")
        # 后2句
        for j in range(idx + 1, min(len(results), idx + 3)):
            print(f"  后文: {results[j]['text']}")
        print()


def run_filler_analysis(
    sentences: List[Dict[str, Any]],
    filler_core: Optional[List[str]] = None,
    filler_extended: Optional[List[str]] = None,
) -> None:
    """执行完整的口癖上下文分析并打印结果。

    Args:
        sentences: ASR 句子列表
        filler_core: 核心口癖词列表
        filler_extended: 扩展口癖词列表
    """
    results = analyze_filler_context(sentences, filler_core, filler_extended)

    print(f"Speaker 总句数: {len(results)}")
    print(f"总字数: {sum(r['char_count'] for r in results)}")

    # 1. 口癖密度最高的句子
    print_high_filler_sentences(results)

    # 2. 口癖Combo句子
    print_combo_sentences(results)

    # 3. 句式模式分析
    print_pattern_analysis(results)

    # 4. 时间段口癖密度
    print_time_segment_density(results)

    # 5. 高口癖句子的上下文
    print_high_filler_context(results)


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════
def main() -> None:
    """命令行入口：解析参数并执行口癖上下文分析。"""
    parser = argparse.ArgumentParser(description="口癖上下文分析工具")
    parser.add_argument("--asr-json", required=True, help="ASR diarization JSON 路径")
    parser.add_argument("--speaker", type=int, default=0, help="目标说话人 ID")
    args = parser.parse_args()

    print(f"[FillerContext] Loading ASR from {args.asr_json}...")
    sentences = load_asr_sentences(args.asr_json, args.speaker)
    print(f"  Loaded {len(sentences)} sentences for speaker {args.speaker}")

    run_filler_analysis(sentences)


if __name__ == "__main__":
    main()
