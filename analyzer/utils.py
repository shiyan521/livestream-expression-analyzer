"""
共享工具模块。

包含全模块共用的常量、工具函数和配置加载逻辑。
"""

import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 全局常量
# ═══════════════════════════════════════════════════════════════════════════════
SR = 22050       # 采样率
HOP = 512        # hop_length

# ─── 口癖词库默认值（当 config/fillers.yaml 不存在时使用） ───────────────────────
DEFAULT_FILLER_CORE: List[str] = [
    "呃", "嗯", "那个", "对对", "这个这个", "就是说",
    "或者说", "嘛", "呢", "吧", "到底",
    "比如说", "来去", "这样一个", "样一个",
]

DEFAULT_FILLER_EXTENDED: List[str] = [
    "这个", "就是", "然后", "其实", "那", "可能", "所以", "确实", "比较",
    "啊", "OK",
    "你可以", "你觉得", "的话", "还是",
]

DEFAULT_TOPIC_MARKERS: List[str] = [
    "下一个话题", "那我们聊", "接下来", "说到", "聊一下", "我们聊",
    "下一个问题", "接下来我们", "那我们", "继续", "换个话题",
    "咱们接着", "可以聊", "那我们可以", "咱们可以",
    "接着下一个", "那聊", "下面", "好呀那我们", "那我们就",
    "简单开个场", "收个尾", "简单的收尾", "感谢", "今天就",
    "自我介绍", "介绍一下",
]

# ─── 说话人识别默认配置（当 config/speakers.yaml 不存在时使用） ─────────────────
DEFAULT_SPEAKER_CONFIG: Dict[str, Any] = {
    "target_speaker": {
        "name": "Speaker A",
        "unique_keywords": [],
        "general_keywords": [],
        "reverse_keywords": [],
    }
}

# ─── 话题标签默认规则（当 config/topic_labels.yaml 不存在时使用） ───────────────
DEFAULT_TOPIC_LABEL_RULES: List[Dict[str, Any]] = []


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_time(sec: Union[int, float]) -> str:
    """秒数格式化为 MM:SS 字符串。

    Args:
        sec: 秒数。

    Returns:
        格式化后的时间字符串，如 "05:30"。
    """
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def safe_float(v: Any, digits: int = 3) -> float:
    """安全浮点数转换，处理 None 和 NaN。

    Args:
        v: 待转换的值。
        digits: 保留小数位数。

    Returns:
        转换后的浮点数，None/NaN 返回 0.0。
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return round(float(v), digits)


def run_cmd(
    cmd: List[str],
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """执行外部命令。

    Args:
        cmd: 命令参数列表。
        check: 为 True 时命令失败则退出。
        capture: 为 True 时捕获输出。

    Returns:
        subprocess.CompletedProcess 对象。
    """
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if check and r.returncode != 0:
        print(
            f"  CMD FAIL: {' '.join(cmd[:6])}...\n  {r.stderr[:300]}",
            file=sys.stderr,
        )
        sys.exit(1)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════════════════════
def _get_config_dir(config_path: Optional[str] = None) -> str:
    """获取配置文件目录。

    优先使用传入的 config_path，否则相对于本模块位置查找 ../config/。

    Args:
        config_path: 可选的配置目录路径。

    Returns:
        配置目录的绝对路径。
    """
    if config_path:
        return os.path.abspath(config_path)
    # 本模块位于 livestream-expression-analyzer/analyzer/utils.py
    # 配置目录位于 livestream-expression-analyzer/config/
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(module_dir)
    return os.path.join(project_dir, "config")


def _load_yaml(filepath: str) -> Optional[Dict[str, Any]]:
    """加载 YAML 文件，失败时返回 None。"""
    try:
        import yaml
    except ImportError:
        print(f"  警告: PyYAML 未安装，无法加载 {filepath}，使用默认值", file=sys.stderr)
        return None

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_filler_config(config_path: Optional[str] = None) -> Dict[str, List[str]]:
    """加载口癖词库配置。

    从 config/fillers.yaml 读取 core_fillers、extended_fillers、topic_markers。
    如果配置文件不存在或读取失败，使用代码中的默认值。

    Args:
        config_path: 可选的配置目录路径。

    Returns:
        包含以下键的字典:
            - core_fillers: 核心口癖列表
            - extended_fillers: 扩展口癖列表
            - topic_markers: 话题标记列表
            - all_fillers: 核心+扩展的合并列表（兼容旧引用）
    """
    config_dir = _get_config_dir(config_path)
    data = _load_yaml(os.path.join(config_dir, "fillers.yaml"))

    if data is None:
        print("  使用默认口癖词库（未找到 config/fillers.yaml）")
        core = DEFAULT_FILLER_CORE
        extended = DEFAULT_FILLER_EXTENDED
        markers = DEFAULT_TOPIC_MARKERS
    else:
        core = data.get("core_fillers", DEFAULT_FILLER_CORE)
        extended = data.get("extended_fillers", DEFAULT_FILLER_EXTENDED)
        markers = data.get("topic_markers", DEFAULT_TOPIC_MARKERS)

    return {
        "core_fillers": core,
        "extended_fillers": extended,
        "topic_markers": markers,
        "all_fillers": core + extended,  # 兼容旧引用 FILLER_WORDS_ZH
    }


def load_speaker_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载说话人识别配置。

    从 config/speakers.yaml 读取 target_speaker 的关键词配置。
    如果配置文件不存在或读取失败，使用默认空配置。

    Args:
        config_path: 可选的配置目录路径。

    Returns:
        说话人配置字典，结构同 DEFAULT_SPEAKER_CONFIG。
    """
    config_dir = _get_config_dir(config_path)
    data = _load_yaml(os.path.join(config_dir, "speakers.yaml"))

    if data is None:
        print("  使用默认说话人配置（未找到 config/speakers.yaml）")
        return DEFAULT_SPEAKER_CONFIG

    return data


def load_topic_label_config(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载话题标签规则配置。

    从 config/topic_labels.yaml 读取规则列表。
    每条规则结构: {"keywords": [...], "label": "..."}。
    如果配置文件不存在或读取失败，返回空列表（使用通用标签 "话题N"）。

    Args:
        config_path: 可选的配置目录路径。

    Returns:
        话题标签规则列表。
    """
    config_dir = _get_config_dir(config_path)
    data = _load_yaml(os.path.join(config_dir, "topic_labels.yaml"))

    if data is None:
        return DEFAULT_TOPIC_LABEL_RULES

    return data.get("rules", [])
