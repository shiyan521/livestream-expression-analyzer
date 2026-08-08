#!/usr/bin/env python3
"""
合并分片 ASR JSON 文件为完整 diarization JSON。

当音频时长超过 ASR 服务的限制时，需要将音频分片处理，
每片生成一个 ASR JSON，最后用本工具合并。

分片文件命名规则：chunk_00_asr.json, chunk_01_asr.json, ...
（位于 --input-dir 指定的目录下）

使用方式：
    # 基本用法（自动推断分片数）
    python tools/merge_asr.py \\
        --input-dir ./asr_chunks \\
        --output merged.json \\
        --offsets 0,2700,5400

    # 指定输入目录和输出文件
    python tools/merge_asr.py \\
        --input-dir /path/to/chunks \\
        --output /path/to/merged.json \\
        --offsets 0,2700,5400,8100

参数说明：
    --input-dir: 包含分片 ASR JSON 文件的目录（必填）
    --output:    合并后的输出 JSON 文件路径（必填）
    --offsets:   每个分片的起始时间偏移量（秒），逗号分隔
                 例如: 0,2700,5400 表示第1片从0秒开始，第2片从2700秒开始...
                 如不提供，将自动按等间距计算
"""

import argparse
import json
import os
import sys

# 默认分片时长（秒），与 analyze.py 保持一致
DEFAULT_CHUNK_MIN = 45 * 60  # 45分钟


def get_chunk_files(input_dir):
    """获取输入目录中所有分片文件，按序号排序。"""
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("chunk_") and fname.endswith("_asr.json"):
            files.append(os.path.join(input_dir, fname))

    if not files:
        print(f"错误: 在 {input_dir} 中未找到分片文件（期望格式: chunk_XX_asr.json）", file=sys.stderr)
        sys.exit(1)

    return files


def parse_offsets(offsets_str, n_chunks):
    """
    解析偏移量参数。

    参数:
        offsets_str: 逗号分隔的偏移量字符串，如 "0,2700,5400"
        n_chunks:    分片数量

    返回:
        偏移量列表（秒）
    """
    if offsets_str:
        try:
            offsets = [float(x.strip()) for x in offsets_str.split(",")]
        except ValueError as e:
            print(f"错误: 无法解析偏移量: {e}", file=sys.stderr)
            sys.exit(1)

        if len(offsets) != n_chunks:
            print(
                f"错误: 偏移量数量({len(offsets)})与分片数({n_chunks})不匹配",
                file=sys.stderr,
            )
            sys.exit(1)
        return offsets

    # 未提供偏移量时，自动按等间距计算
    print("提示: 未提供 --offsets，将自动按等间距计算偏移量", file=sys.stderr)
    # 由于无法获取音频时长，这里使用默认分片时长
    return [i * DEFAULT_CHUNK_MIN for i in range(n_chunks)]


def merge(input_dir, output_file, offsets):
    """
    合并分片 ASR JSON 文件。

    参数:
        input_dir:   分片文件所在目录
        output_file: 合并后输出文件路径
        offsets:     每个分片的起始时间偏移量列表（秒）

    返回:
        合并后的数据字典
    """
    chunk_files = get_chunk_files(input_dir)
    n_chunks = len(chunk_files)

    if len(offsets) != n_chunks:
        offsets = parse_offsets(None, n_chunks)

    print(f"找到 {n_chunks} 个分片文件")
    print(f"偏移量(秒): {offsets}")

    all_sentences = []
    max_end_time = 0

    for i, chunk_file in enumerate(chunk_files):
        chunk_offset_ms = int(offsets[i] * 1000)

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        chunk_sents = []
        for t in chunk_data.get("transcripts", []):
            for s in t.get("sentences", []):
                begin = s["begin_time"] + chunk_offset_ms
                end = s["end_time"] + chunk_offset_ms
                chunk_sents.append({
                    "begin_time": begin,
                    "end_time": end,
                    "text": s["text"],
                    "speaker_id": s["speaker_id"],
                    "words": s.get("words", []),
                })
                if end > max_end_time:
                    max_end_time = end

        all_sentences.extend(chunk_sents)
        print(f"  切片 {i + 1}/{n_chunks}: {len(chunk_sents)} 句 (偏移: {offsets[i]:.0f}s)")

    # 按时间排序
    all_sentences.sort(key=lambda x: x["begin_time"])

    # 构建合并后的数据
    merged_data = {
        "transcripts": [{"sentences": all_sentences}],
        "properties": {
            "original_duration_in_milliseconds": max_end_time,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成: {len(all_sentences)} 句 -> {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024:.0f} KB")
    print(f"总时长: {max_end_time}ms ({max_end_time / 1000:.1f}s)")

    return merged_data


def main():
    parser = argparse.ArgumentParser(
        description="合并分片 ASR JSON 文件为完整 diarization JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python tools/merge_asr.py -d ./asr_chunks -o merged.json \\
      --offsets 0,2700,5400

  # 自动计算偏移量（等间距）
  python tools/merge_asr.py -d ./asr_chunks -o merged.json
        """,
    )
    parser.add_argument(
        "--input-dir", "-d",
        required=True,
        help="包含分片 ASR JSON 文件的目录",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="合并后的输出 JSON 文件路径",
    )
    parser.add_argument(
        "--offsets",
        default=None,
        help="每个分片的起始时间偏移量（秒），逗号分隔。"
             "例如: 0,2700,5400。未提供时将自动按等间距计算。",
    )

    args = parser.parse_args()

    chunk_files = get_chunk_files(args.input_dir)
    n_chunks = len(chunk_files)
    offsets = parse_offsets(args.offsets, n_chunks) if args.offsets else \
        [i * DEFAULT_CHUNK_MIN for i in range(n_chunks)]

    merge(args.input_dir, args.output, offsets)


if __name__ == "__main__":
    main()
