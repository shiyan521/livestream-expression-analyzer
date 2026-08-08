#!/usr/bin/env python3
"""
阿里云百炼ASR后端（可选，付费）。

通过 subprocess 调用 `bl` CLI 做语音转写 + 说话人分离。
音频超过45分钟时自动分片处理以避免超时。

前置条件:
  1. 安装 bl CLI（阿里云百炼 CLI）
  2. 配置 API Key（运行 bl auth login 或设置环境变量）
  3. 系统中需有 ffmpeg（分片时使用）

安装 bl CLI:
  pip install bl
  # 或参考阿里云百炼官方文档

用法示例:
  backend = BailianBackend()
  result = backend.transcribe("/path/to/audio.mp3", speaker_count=2)
"""
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .base import ASRBackend

# 分片阈值：45分钟（秒）
CHUNK_THRESHOLD_SECONDS: int = 45 * 60
# 默认采样率（用于检测音频时长）
DEFAULT_SAMPLE_RATE: int = 22050


def _run_cmd(
    cmd: List[str],
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """运行命令行并返回结果。

    Args:
        cmd: 命令列表
        check: 是否在非零退出码时抛出异常
        capture: 是否捕获输出

    Returns:
        CompletedProcess 实例

    Raises:
        SystemExit: 当 check=True 且命令失败时
    """
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if check and r.returncode != 0:
        print(f"  CMD FAIL: {' '.join(cmd[:6])}...\n  {r.stderr[:300]}", file=sys.stderr)
        sys.exit(1)
    return r


def _load_asr_json(json_path: str) -> Tuple[List[Dict[str, Any]], float]:
    """加载ASR JSON文件，返回句子列表和总时长。

    Args:
        json_path: ASR JSON 文件路径

    Returns:
        (sentences, duration_seconds) 元组
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
    duration_ms = data.get("properties", {}).get("original_duration_in_milliseconds", 0)
    return sentences, duration_ms / 1000.0 if duration_ms else 0.0


class BailianBackend(ASRBackend):
    """阿里云百炼ASR后端。

    通过调用 `bl` CLI 实现语音转写和说话人分离。
    音频超过45分钟时自动分片处理。

    Attributes:
        bl_path: bl CLI 的可执行路径，默认为 'bl'（需在 PATH 中）
        timeout: 单次ASR请求超时时间（秒）
        sample_rate: 用于检测音频时长的采样率
    """

    def __init__(
        self,
        bl_path: str = "bl",
        timeout: int = 300,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """初始化百炼后端。

        Args:
            bl_path: bl CLI 可执行路径，默认 'bl'
            timeout: 单次ASR请求超时秒数，默认 300
            sample_rate: 检测音频时长的采样率，默认 22050
        """
        self.bl_path = bl_path
        self.timeout = timeout
        self.sample_rate = sample_rate

    def _get_audio_duration(self, audio_path: str) -> float:
        """检测音频时长（秒）。

        Args:
            audio_path: 音频文件路径

        Returns:
            音频时长（秒）
        """
        import librosa
        y_tmp, sr_tmp = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(y_tmp) / sr_tmp
        del y_tmp, sr_tmp
        return duration

    def _run_single_asr(
        self,
        audio_path: str,
        speaker_count: int,
        output_json: str,
    ) -> str:
        """单次ASR转写（不分片）。

        Args:
            audio_path: 音频文件路径
            speaker_count: 说话人数量
            output_json: 输出 JSON 路径

        Returns:
            输出 JSON 文件路径
        """
        cmd = [
            self.bl_path, "speech", "recognize",
            "--diarization",
            "--speaker-count", str(speaker_count),
            "--url", audio_path,
            "--out", output_json,
            "--timeout", str(self.timeout),
        ]
        _run_cmd(cmd)
        return output_json

    def _run_chunked_asr(
        self,
        audio_path: str,
        speaker_count: int,
        output_json: str,
        audio_duration: float,
    ) -> str:
        """分片ASR转写（音频>45分钟时使用）。

        将音频按时间均分为多个片段，分别做ASR，
        然后合并结果并偏移时间戳。

        Args:
            audio_path: 音频文件路径
            speaker_count: 说话人数量
            output_json: 最终输出 JSON 路径
            audio_duration: 音频总时长（秒）

        Returns:
            输出 JSON 文件路径
        """
        n_chunks = int(audio_duration // CHUNK_THRESHOLD_SECONDS) + 1
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
            _run_cmd(cmd, check=False)
            if not os.path.exists(chunk_file):
                print(f"  警告: 切片 {i} 失败，跳过", file=sys.stderr)
                continue

            # ASR
            cmd = [
                self.bl_path, "speech", "recognize",
                "--diarization",
                "--speaker-count", str(speaker_count),
                "--url", chunk_file,
                "--out", chunk_asr,
                "--timeout", str(self.timeout),
            ]
            _run_cmd(cmd)

            # 加载并偏移时间戳
            chunk_sents, _ = _load_asr_json(chunk_asr)
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
        return output_json

    def transcribe(self, audio_path: str, speaker_count: int = 2) -> Dict[str, Any]:
        """转写音频并做说话人分离。

        音频超过45分钟时自动分片处理。

        Args:
            audio_path: 音频文件路径
            speaker_count: 说话人数量

        Returns:
            ASR diarization JSON格式
        """
        print(f"[BailianBackend] ASR + 说话人分离 (speakers={speaker_count})...")

        # 检测音频时长
        audio_duration = self._get_audio_duration(audio_path)
        print(f"  音频时长: {audio_duration:.0f}s ({audio_duration/60:.1f}min)")

        # 使用临时文件存储结果
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="bailian_asr_"
        ) as tmp:
            output_json = tmp.name

        try:
            if audio_duration <= CHUNK_THRESHOLD_SECONDS:
                # 正常单次ASR
                self._run_single_asr(audio_path, speaker_count, output_json)
            else:
                # 分片ASR
                self._run_chunked_asr(audio_path, speaker_count, output_json, audio_duration)

            # 读取结果
            with open(output_json, "r", encoding="utf-8") as f:
                result = json.load(f)

            return result
        finally:
            # 清理临时文件
            if os.path.exists(output_json):
                os.unlink(output_json)
