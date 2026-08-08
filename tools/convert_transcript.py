#!/usr/bin/env python3
"""
将逐字稿文本文件转换为 ASR diarization JSON 格式。

支持多种逐字稿输入格式，输出统一的 diarization JSON 供分析管线使用。

输入格式（每行一段）：
    说话人(HH:MM:SS): 文字内容

输出格式：ASR diarization JSON，包含 transcripts 和 properties 字段。

使用方式：
    # 基本用法
    python tools/convert_transcript.py \\
        --input transcript.txt \\
        --output output.json

    # 指定说话人映射
    python tools/convert_transcript.py \\
        --input transcript.txt \\
        --output output.json \\
        --speaker-map '{"Alice": 0, "Bob": 1}'

    # 从文件加载说话人映射
    python tools/convert_transcript.py \\
        --input transcript.txt \\
        --output output.json \\
        --speaker-map speaker_map.json

参数说明：
    --input:       输入逐字稿文件路径（必填）
    --output:      输出 JSON 文件路径（必填）
    --speaker-map: 说话人映射，JSON 字符串或 JSON 文件路径
                   格式: {"说话人名称": speaker_id, ...}
                   未映射的说话人将被跳过并警告
"""

import argparse
import json
import re
import sys

# 匹配逐字稿行的正则: 说话人(HH:MM:SS): 文字内容
LINE_PATTERN = re.compile(r"^(.+?)\((\d{2}:\d{2}:\d{2})\):\s*(.*)")


def time_to_ms(time_str: str) -> int:
    """将 HH:MM:SS 时间字符串转换为毫秒。"""
    h, m, s = time_str.split(":")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000


def load_speaker_map(speaker_map_arg: str) -> dict:
    """
    加载说话人映射。

    参数可以是 JSON 字符串或 JSON 文件路径。
    如果未提供，返回空字典（所有说话人将被自动编号）。
    """
    if not speaker_map_arg:
        return {}

    # 尝试作为文件路径加载
    try:
        with open(speaker_map_arg, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # 尝试作为 JSON 字符串解析
    try:
        return json.loads(speaker_map_arg)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 --speaker-map 参数: {e}", file=sys.stderr)
        print("请提供有效的 JSON 字符串或 JSON 文件路径", file=sys.stderr)
        sys.exit(1)


def convert(input_file: str, output_file: str, speaker_map: dict):
    """
    将逐字稿文本转换为 diarization JSON。

    参数:
        input_file:  输入逐字稿文件路径
        output_file: 输出 JSON 文件路径
        speaker_map: 说话人名称到 speaker_id 的映射

    返回:
        转换后的数据字典
    """
    sentences = []
    max_end_time = 0

    # 如果未提供说话人映射，自动编号
    auto_speaker_map = {}
    next_speaker_id = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = LINE_PATTERN.match(line)
        if not match:
            print(f"  [WARN] 无法解析行，跳过: {line[:80]}...", file=sys.stderr)
            continue

        speaker_name = match.group(1)
        time_str = match.group(2)
        text = match.group(3).strip()

        if not text:
            # 时间戳行但无文字内容，跳过
            continue

        # 确定 speaker_id
        if speaker_map:
            speaker_id = speaker_map.get(speaker_name)
            if speaker_id is None:
                print(f"  [WARN] 未知说话人 '{speaker_name}'，跳过: {line[:80]}...", file=sys.stderr)
                continue
        else:
            # 自动编号
            if speaker_name not in auto_speaker_map:
                auto_speaker_map[speaker_name] = next_speaker_id
                next_speaker_id += 1
            speaker_id = auto_speaker_map[speaker_name]

        begin_time = time_to_ms(time_str)
        # 结束时间暂时设为与 begin_time 相同，后续根据下一条修正
        sentences.append({
            "begin_time": begin_time,
            "end_time": begin_time,  # 占位，稍后修正
            "text": text,
            "speaker_id": speaker_id,
            "words": [],
        })

    if not sentences:
        print("错误: 未找到有效的逐字稿内容", file=sys.stderr)
        sys.exit(1)

    # 修正 end_time：每句的 end_time 设为下一句的 begin_time
    # 最后一句的 end_time 设为 begin_time + 1000ms（保守估计至少1秒）
    for i in range(len(sentences)):
        if i < len(sentences) - 1:
            sentences[i]["end_time"] = sentences[i + 1]["begin_time"]
        else:
            sentences[i]["end_time"] = sentences[i]["begin_time"] + 1000

    # 计算总时长
    if sentences:
        max_end_time = sentences[-1]["end_time"]
    original_duration = max_end_time + 1000  # 加1秒余量

    # 构建输出结构
    output = {
        "transcripts": [
            {
                "sentences": sentences,
            }
        ],
        "properties": {
            "original_duration_in_milliseconds": original_duration,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    final_speaker_map = speaker_map if speaker_map else auto_speaker_map
    print(f"\n转换完成！")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  总行数: {len(lines)}")
    print(f"  有效句子数: {len(sentences)}")
    print(f"  说话人映射: {final_speaker_map}")
    print(f"  最后一条时间戳: {max_end_time}ms")
    print(f"  音频总时长(估计): {original_duration}ms ({original_duration/1000:.1f}s)")

    # 统计各说话人句数
    speaker_counts = {}
    for s in sentences:
        sid = s["speaker_id"]
        speaker_counts[sid] = speaker_counts.get(sid, 0) + 1
    print(f"  各说话人句数: {speaker_counts}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="将逐字稿文本文件转换为 ASR diarization JSON 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python tools/convert_transcript.py -i transcript.txt -o output.json

  # 指定说话人映射（JSON 字符串）
  python tools/convert_transcript.py -i transcript.txt -o output.json \\
      --speaker-map '{"Alice": 0, "Bob": 1}'

  # 从文件加载说话人映射
  python tools/convert_transcript.py -i transcript.txt -o output.json \\
      --speaker-map speaker_map.json
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入逐字稿文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--speaker-map", "-s",
        default=None,
        help="说话人映射，JSON 字符串或 JSON 文件路径。"
             '格式: {"说话人名称": speaker_id}。'
             "未提供时将自动编号。",
    )

    args = parser.parse_args()

    speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else {}
    convert(args.input, args.output, speaker_map)


if __name__ == "__main__":
    main()
