#!/usr/bin/env python3
"""
Livestream Expression Analyzer - Web 服务

提供音频上传、ASR 转写、表达分析及报告展示功能。

支持的 ASR 后端：
  - whisper:   使用 openai-whisper 本地转写（免费）
  - transcript: 用户上传逐字稿文件导入（最准确，推荐）
  - bailian:   阿里云百炼 ASR 服务（需自行配置 API Key）

分析管线已接入 analyzer.pipeline：上传文件后将执行真实的多维度分析，
分析失败或依赖缺失时回退到内置 Demo 数据。

启动方式：
    python web/app.py
默认端口：5000
"""

import json
import os
import sys
import tempfile
import time

from flask import Flask, jsonify, request, send_from_directory

# 将项目根目录加入 sys.path，确保 analyzer / asr 包可导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analyzer.pipeline import analyze as pipeline_analyze, load_asr_json  # noqa: E402
from web.report_adapter import to_report_view  # noqa: E402

app = Flask(__name__, static_folder="static")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 支持的音频/视频格式
ALLOWED_AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".mp4", ".webm"}
# 支持的逐字稿格式
ALLOWED_TRANSCRIPT_EXTS = {".json", ".txt", ".docx", ".srt"}

# 加载 Demo 数据
with open(os.path.join(STATIC_DIR, "demo-data.json"), "r", encoding="utf-8") as f:
    DEMO_DATA = json.load(f)


def check_whisper_available():
    """检查 openai-whisper 是否已安装。"""
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def check_bailian_available():
    """检查阿里云百炼 SDK 是否已安装并配置。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    return bool(api_key)


def run_whisper_pipeline(audio_path):
    """
    使用 Whisper 本地转写 + 说话人分离，然后调用分析管线生成报告。

    返回报告视图 dict 或 None（如果依赖缺失或分析失败）。
    """
    try:
        from asr.whisper_backend import WhisperBackend
    except ImportError:
        return None

    return _run_analyze(audio_path, WhisperBackend(), "whisper")


def run_transcript_pipeline(audio_path, transcript_path):
    """
    使用用户上传的逐字稿 + 音频进行真实分析。

    逐字稿作为说话人分离与时间戳的 ground truth（最高准确度模式），
    音频用于声学特征提取。支持腾讯会议段落式、飞书妙记、
    标准单行及 JSON 四种逐字稿格式，自动检测。

    返回报告视图 dict 或 None（解析/分析失败时）。
    """
    try:
        from asr.transcript_import import TranscriptImportBackend
    except ImportError:
        return None

    with tempfile.TemporaryDirectory() as workdir:
        try:
            # 1. 逐字稿 → diarization JSON（说话人自动编号）
            backend = TranscriptImportBackend()
            diar = backend.transcribe(transcript_path)
            if not diar["transcripts"][0]["sentences"]:
                print("[transcript] 逐字稿解析失败：未找到有效句子")
                return None

            asr_json_path = os.path.join(workdir, "transcript_asr.json")
            with open(asr_json_path, "w", encoding="utf-8") as f:
                json.dump(diar, f, ensure_ascii=False, indent=2)

            # 2. 统计实际说话人数量
            sentences, _ = load_asr_json(asr_json_path)
            speaker_count = len({s["speaker"] for s in sentences}) or 2

            # 3. 调用主分析管线（跳过 ASR，逐字稿即 ground truth）
            result = pipeline_analyze(
                audio_path=audio_path,
                asr_backend=backend,
                speakers=speaker_count,
                target_speaker="auto",
                output_path=os.path.join(workdir, "result.json"),
                skip_asr=True,
                asr_json=asr_json_path,
                skip_clips=True,
                session_id="web",
                enable_content_analysis=True,
                llm_backend_name="auto",
            )
            return to_report_view(result)
        except Exception as e:
            print(f"[transcript] 分析失败: {e}")
            return None


def run_bailian_pipeline(audio_path):
    """
    使用阿里云百炼 ASR 服务进行转写和分析。
    需要 DASHSCOPE_API_KEY 环境变量。

    返回报告视图 dict 或 None。
    """
    if not check_bailian_available():
        return None
    try:
        from asr.bailian_backend import BailianBackend
    except ImportError:
        return None

    return _run_analyze(audio_path, BailianBackend(), "bailian")


def _run_analyze(audio_path, asr_backend, backend_name):
    """通用分析入口：ASR 转写 + 分析管线 + 报告视图转换。"""
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = pipeline_analyze(
                audio_path=audio_path,
                asr_backend=asr_backend,
                speakers=2,
                target_speaker="auto",
                output_path=os.path.join(workdir, "result.json"),
                skip_clips=True,
                session_id="web",
                enable_content_analysis=True,
                llm_backend_name="auto",
            )
            return to_report_view(result)
        except Exception as e:
            print(f"[{backend_name}] 分析失败: {e}")
            return None


@app.route("/")
def index():
    """首页 - 上传入口。"""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/report")
def report_page():
    """报告页 - 展示分析结果。"""
    return send_from_directory(STATIC_DIR, "report.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    分析接口

    请求参数（multipart/form-data）：
      - file:        音频/视频文件（必填）
      - transcript:  逐字稿文件（可选，当 asr_backend=transcript 时必填）
      - asr_backend: ASR 后端选择（可选，默认 whisper）
                     可选值：whisper / transcript / bailian

    返回：
      - 成功：{ success, session_id, data, asr_backend, is_demo }
      - 失败：{ error }
      - ASR 后端不可用：{ success, session_id, data, is_demo, demo_reason }
    """
    # ---- 校验音频文件 ----
    if "file" not in request.files:
        return jsonify({"error": "未找到音频文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        return jsonify({
            "error": f"不支持的格式: {ext}，仅支持 {', '.join(sorted(ALLOWED_AUDIO_EXTS))}"
        }), 400

    # ---- 读取 ASR 后端选择 ----
    asr_backend = request.form.get("asr_backend", "whisper").lower()

    # ---- 处理逐字稿文件 ----
    transcript_file = request.files.get("transcript")
    transcript_path = None
    if transcript_file and transcript_file.filename:
        t_ext = os.path.splitext(transcript_file.filename)[1].lower()
        if t_ext not in ALLOWED_TRANSCRIPT_EXTS:
            return jsonify({
                "error": f"不支持的逐字稿格式: {t_ext}，仅支持 {', '.join(sorted(ALLOWED_TRANSCRIPT_EXTS))}"
            }), 400
        # 保存到临时文件
        t_fd, transcript_path = tempfile.mkstemp(suffix=t_ext)
        transcript_file.save(transcript_path)
        os.close(t_fd)

    # ---- 保存音频到临时文件 ----
    a_fd, audio_path = tempfile.mkstemp(suffix=ext)
    file.save(audio_path)
    os.close(a_fd)

    try:
        # ---- 根据 ASR 后端调用分析管线 ----
        result = None
        demo_reason = ""

        if asr_backend == "transcript":
            if not transcript_path:
                return jsonify({
                    "error": "选择'逐字稿导入'时必须上传逐字稿文件"
                }), 400
            result = run_transcript_pipeline(audio_path, transcript_path)
            if result is None:
                demo_reason = "逐字稿分析管线尚未完成配置，已返回 Demo 数据"

        elif asr_backend == "bailian":
            if not check_bailian_available():
                demo_reason = "阿里云百炼 ASR 后端未配置（需设置 DASHSCOPE_API_KEY 环境变量），已返回 Demo 数据"
            else:
                result = run_bailian_pipeline(audio_path)
                if result is None:
                    demo_reason = "百炼 ASR 分析管线尚未完成配置，已返回 Demo 数据"

        elif asr_backend == "whisper":
            if not check_whisper_available():
                demo_reason = "Whisper ASR 后端未安装（请运行 pip install openai-whisper），已返回 Demo 数据"
            else:
                result = run_whisper_pipeline(audio_path)
                if result is None:
                    demo_reason = "Whisper 分析管线尚未完成配置，已返回 Demo 数据"

        else:
            return jsonify({
                "error": f"未知的 ASR 后端: {asr_backend}，可选值: whisper / transcript / bailian"
            }), 400

        # ---- 返回结果 ----
        if result is not None:
            # 真实分析成功
            return jsonify({
                "success": True,
                "session_id": result.get("meta", {}).get("session_id", "session"),
                "data": result,
                "asr_backend": asr_backend,
                "is_demo": False
            })
        else:
            # ASR 后端不可用或管线未完成，返回 Demo 数据并提示
            return jsonify({
                "success": True,
                "session_id": DEMO_DATA["meta"]["session_id"],
                "data": DEMO_DATA,
                "asr_backend": asr_backend,
                "is_demo": True,
                "demo_reason": demo_reason or "分析管线尚未完成配置，已返回 Demo 数据"
            })

    finally:
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if transcript_path and os.path.exists(transcript_path):
            os.remove(transcript_path)


@app.route("/api/demo")
def demo_data():
    """返回 Demo 数据，供报告页直接展示。"""
    return jsonify({"success": True, "data": DEMO_DATA})


@app.route("/api/health")
def health():
    """健康检查接口。"""
    return jsonify({
        "status": "ok",
        "whisper_available": check_whisper_available(),
        "bailian_available": check_bailian_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/static/<path:path>")
def static_files(path):
    """静态文件服务。"""
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
