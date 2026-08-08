#!/usr/bin/env python3
"""
BL CLI ASR 后端实现。

通过调用 `bl speech recognize` 命令行工具完成语音识别 + 说话人分离。
这是从原 analyze.py 中提取的默认后端实现，与 BailianBackend 功能类似但更轻量
（不含自动分片逻辑，分片由 pipeline.run_asr 处理）。
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

from .base import ASRBackend


class BLASRBackend(ASRBackend):
    """基于 bl CLI 的 ASR 后端实现。

    调用 `bl speech recognize --diarization` 完成语音识别 + 说话人分离。

    与 BailianBackend 的区别:
        - 不含自动分片逻辑（分片由上层 pipeline.run_asr 处理）
        - recognize() 直接使用 bl CLI 的 --out 参数输出到指定文件
        - 更轻量，适合作为默认后端使用
    """

    def transcribe(
        self,
        audio_path: str,
        speaker_count: int = 2,
    ) -> Dict[str, Any]:
        """调用 bl CLI 执行 ASR + 说话人分离，返回结果字典。

        Args:
            audio_path: 音频文件路径。
            speaker_count: 预期说话人数量。

        Returns:
            ASR diarization JSON 格式的字典。

        Raises:
            SystemExit: bl 命令执行失败时退出。
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="bl_asr_"
        ) as tmp:
            tmp_path = tmp.name

        try:
            self._run_bl_cli(audio_path, speaker_count, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                return json.load(f)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def recognize(
        self,
        audio_path: str,
        speaker_count: int,
        output_json: str,
        timeout: int = 300,
    ) -> str:
        """调用 bl CLI 执行 ASR + 说话人分离，直接输出到指定文件。

        覆盖基类方法，直接使用 bl CLI 的 --out 参数，
        避免 transcribe() 中不必要的中间文件读写。

        Args:
            audio_path: 音频文件路径。
            speaker_count: 预期说话人数量。
            output_json: 输出 JSON 文件路径。
            timeout: 超时时间（秒）。

        Returns:
            输出 JSON 文件路径。

        Raises:
            SystemExit: bl 命令执行失败时退出。
        """
        self._run_bl_cli(audio_path, speaker_count, output_json, timeout)
        return output_json

    def _run_bl_cli(
        self,
        audio_path: str,
        speaker_count: int,
        output_json: str,
        timeout: int = 300,
    ) -> None:
        """执行 bl speech recognize 命令。

        Args:
            audio_path: 音频文件路径。
            speaker_count: 预期说话人数量。
            output_json: 输出 JSON 文件路径。
            timeout: 超时时间（秒）。

        Raises:
            SystemExit: 命令执行失败时退出。
        """
        cmd: List[str] = [
            "bl", "speech", "recognize",
            "--diarization",
            "--speaker-count", str(speaker_count),
            "--url", audio_path,
            "--out", output_json,
            "--timeout", str(timeout),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(
                f"  CMD FAIL: {' '.join(cmd[:6])}...\n  {r.stderr[:300]}",
                file=sys.stderr,
            )
            sys.exit(1)
