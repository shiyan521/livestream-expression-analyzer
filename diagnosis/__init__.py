"""
diagnosis 模块 - 直播音频发音诊断。

提供七个维度的深层发音分析能力：
  - 尾音消失检测
  - 咬字紧张度
  - 语速-清晰度联动分析
  - 音高稳定性 / Jitter
  - 呼吸节奏分析
  - 长句尾段能量衰减
  - 英文词汇发音检测
"""

from .pronunciation import (
    analysis_1_tail_drop,
    analysis_2_tension,
    analysis_3_speed_clearness,
    analysis_4_pitch_stability,
    analysis_5_breathing,
    analysis_6_long_sentence_tail,
    analysis_7_english_words,
    compute_f0_cv,
    compute_rms,
    compute_rms_over_time,
    compute_spectral_centroid_mean,
    compute_spectral_flatness_mean,
    detect_speaker_id,
    extract_audio_segment,
    find_breaths,
    format_time,
    get_speaker_sentences,
    load_asr_data,
    run_pronunciation_diagnosis,
)

__all__ = [
    'analysis_1_tail_drop',
    'analysis_2_tension',
    'analysis_3_speed_clearness',
    'analysis_4_pitch_stability',
    'analysis_5_breathing',
    'analysis_6_long_sentence_tail',
    'analysis_7_english_words',
    'compute_f0_cv',
    'compute_rms',
    'compute_rms_over_time',
    'compute_spectral_centroid_mean',
    'compute_spectral_flatness_mean',
    'detect_speaker_id',
    'extract_audio_segment',
    'find_breaths',
    'format_time',
    'get_speaker_sentences',
    'load_asr_data',
    'run_pronunciation_diagnosis',
]
