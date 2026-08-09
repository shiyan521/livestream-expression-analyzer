"""
livestream-expression-analyzer: 直播表达分析模块化包。

从单体脚本 analyze.py 重构而来，提供模块化的直播音频全量分析能力。

主要接口:
    - analyze(): 主分析管线函数，接受 ASRBackend 实例
    - ASRBackend: ASR 后端抽象基类
    - BLASRBackend: 默认的 bl CLI ASR 后端实现

子模块:
    - pipeline: 主管线编排（ASR、说话人识别、评分计算）
    - acoustic: 声学特征提取（F0、RMS、频谱、清晰度、非语音事件）
    - text_metrics: 文本指标（口癖、语速、句长、停顿、碎片率）
    - emotion: 情绪曲线与疲劳检测
    - interaction: 互动节奏（轮次、回应、打断、主导度）
    - topics: 话题分割与切片候选
    - dialogue_parser: 对话结构解析（轮次构建、交换对识别、回应类型分类）
    - llm_analyzer: LLM 分析抽象层（多后端支持、内容分析器）
    - content_metrics: 内容层指标（对话接球率、信息增量密度、内容深度、话题连贯性）
    - cognitive_metrics: 认知层指标（论证链追踪、思维深度评估、认知负荷与内容质量关联）
    - utils: 共享工具函数与配置加载

用法示例:
    from analyzer import analyze
    from asr.bl_backend import BLASRBackend

    result = analyze(
        audio_path="/path/to/audio.mp3",
        asr_backend=BLASRBackend(),
        speakers=2,
        output_path="result.json",
    )

    # 自定义 ASR 后端
    from analyzer import analyze
    from asr.base import ASRBackend

    class MyBackend(ASRBackend):
        def recognize(self, audio_path, speaker_count, output_json, timeout=300):
            # 实现自定义 ASR 逻辑
            ...
            return output_json

    result = analyze(
        audio_path="/path/to/audio.mp3",
        asr_backend=MyBackend(),
    )
"""

from analyzer.acoustic import (
    compute_acoustic_features,
    compute_clarity,
    compute_non_speech_events,
)
from analyzer.content_metrics import compute_content_metrics
from analyzer.cognitive_metrics import compute_cognitive_metrics
from analyzer.cross_session import CrossSessionTracker
from analyzer.dialogue_parser import parse_dialogue
from analyzer.emotion import compute_emotion_and_fatigue
from analyzer.interaction import compute_interaction
from analyzer.llm_analyzer import ContentAnalyzer, LLMBackend, get_llm_backend
from analyzer.pipeline import (
    analyze,
    auto_detect_speaker,
    extract_speaker_audio,
    load_asr_json,
    run_asr,
)
from analyzer.report_generator import ReportGenerator
from analyzer.text_metrics import compute_text_metrics
from analyzer.topics import (
    auto_topic_label,
    classify_structure,
    compute_topics_and_clips,
)
from analyzer.utils import (
    SR,
    HOP,
    fmt_time,
    load_filler_config,
    load_speaker_config,
    load_topic_label_config,
    safe_float,
)

__version__ = "1.0.0"

__all__ = [
    # 主管线
    "analyze",
    "run_asr",
    "load_asr_json",
    "auto_detect_speaker",
    "extract_speaker_audio",
    # 声学
    "compute_acoustic_features",
    "compute_clarity",
    "compute_non_speech_events",
    # 文本
    "compute_text_metrics",
    # 情绪
    "compute_emotion_and_fatigue",
    # 互动
    "compute_interaction",
    # 话题
    "compute_topics_and_clips",
    "auto_topic_label",
    "classify_structure",
    # 内容层
    "parse_dialogue",
    "compute_content_metrics",
    "get_llm_backend",
    "ContentAnalyzer",
    "LLMBackend",
    # 认知层
    "compute_cognitive_metrics",
    # 跨场次
    "CrossSessionTracker",
    "ReportGenerator",
    # 工具
    "SR",
    "HOP",
    "fmt_time",
    "safe_float",
    "load_filler_config",
    "load_speaker_config",
    "load_topic_label_config",
    # 版本
    "__version__",
]
