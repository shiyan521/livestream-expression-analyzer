"""
LLM 分析抽象层模块。

提供统一的 LLM 后端接口和内容分析器，支持多种 LLM 提供商。
采用与 ASR 后端相同的抽象模式，实现可插拔的 LLM 后端。

支持的 LLM 后端:
  - dashscope: 阿里云百炼（DashScope）后端（默认，需 DASHSCOPE_API_KEY）
  - openai: OpenAI 兼容后端（支持 OpenAI、Azure OpenAI 等）
  - mock: 模拟后端（用于测试和无需 API 的场景）

用法示例:
  from analyzer.llm_analyzer import get_llm_backend

  # 自动选择可用的后端
  backend = get_llm_backend("auto")
  if backend.available:
      response = backend.chat("You are a helpful assistant.", "Hello!")
  else:
      print("LLM 后端不可用，将使用纯规则分析")

  # 指定后端
  backend = get_llm_backend("dashscope", api_key="your-key", model="qwen-plus")
"""

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from analyzer.utils import safe_float


# ══════════════════════════════════════════════════════════════
# LLM 后端抽象接口
# ══════════════════════════════════════════════════════════════
class LLMBackend(ABC):
    """LLM 后端抽象接口。

    所有 LLM 后端实现都必须继承此类并实现 chat 方法。
    embedding 方法为可选，子类可按需实现。

    属性:
        model: 模型名称
        available: 后端是否可用（已配置 API Key 等）
    """

    @property
    @abstractmethod
    def available(self) -> bool:
        """后端是否可用。"""
        ...

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """发送聊天补全请求。

        Args:
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            temperature: 采样温度，0 表示最确定，1 表示最随机。
            max_tokens: 最大生成 token 数。

        Returns:
            LLM 生成的文本响应。

        Raises:
            RuntimeError: 当后端不可用或请求失败时。
        """
        ...

    def embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量。

        默认实现抛出 NotImplementedError，子类可覆盖。

        Args:
            text: 输入文本。

        Returns:
            嵌入向量（浮点数列表）。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持 embedding 功能"
        )


# ══════════════════════════════════════════════════════════════
# DashScope（阿里云百炼）后端
# ══════════════════════════════════════════════════════════════
class DashScopeBackend(LLMBackend):
    """阿里云百炼（DashScope）LLM 后端。

    使用 dashscope Python SDK 调用通义千问系列模型。
    需要设置环境变量 DASHSCOPE_API_KEY。

    Attributes:
        api_key: DashScope API Key，默认从环境变量读取。
        model: 模型名称，默认 qwen-plus。
        embedding_model: 嵌入模型名称，默认 text-embedding-v2。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        embedding_model: str = "text-embedding-v2",
    ) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.embedding_model = embedding_model
        self._sdk = None

    def _ensure_sdk(self) -> bool:
        """延迟导入 dashscope SDK。

        Returns:
            SDK 是否成功导入。
        """
        if self._sdk is not None:
            return True
        try:
            import dashscope
            self._sdk = dashscope
            if self.api_key:
                dashscope.api_key = self.api_key
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        """后端是否可用（SDK 已安装且 API Key 已配置）。"""
        return self._ensure_sdk() and bool(self.api_key)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """调用 DashScope 聊天补全 API。

        Args:
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。

        Returns:
            LLM 生成的文本。

        Raises:
            RuntimeError: 当后端不可用或请求失败时。
        """
        if not self.available:
            raise RuntimeError(
                "DashScope 后端不可用：请安装 dashscope SDK 并设置 DASHSCOPE_API_KEY"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._sdk.Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format="message",
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise RuntimeError(
                    f"DashScope API 错误: {response.status_code} - {response.message}"
                )
        except Exception as e:
            raise RuntimeError(f"DashScope 请求失败: {e}") from e

    def embedding(self, text: str) -> List[float]:
        """调用 DashScope 嵌入 API。

        Args:
            text: 输入文本。

        Returns:
            嵌入向量。
        """
        if not self.available:
            raise RuntimeError("DashScope 后端不可用")

        try:
            response = self._sdk.TextEmbedding.call(
                model=self.embedding_model,
                input=text,
            )
            if response.status_code == 200:
                return response.output.embeddings[0].embedding
            else:
                raise RuntimeError(
                    f"DashScope Embedding 错误: {response.status_code}"
                )
        except Exception as e:
            raise RuntimeError(f"DashScope Embedding 请求失败: {e}") from e


# ══════════════════════════════════════════════════════════════
# OpenAI 兼容后端
# ══════════════════════════════════════════════════════════════
class OpenAICompatibleBackend(LLMBackend):
    """OpenAI 兼容 LLM 后端。

    支持 OpenAI、Azure OpenAI 以及任何兼容 OpenAI Chat API 的服务。
    使用 openai Python SDK。

    Attributes:
        api_key: API Key，默认从环境变量 OPENAI_API_KEY 读取。
        base_url: API 基础 URL，用于兼容服务。
        model: 模型名称，默认 gpt-4o-mini。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.model = model
        self._client = None

    def _ensure_client(self) -> bool:
        """延迟创建 OpenAI 客户端。"""
        if self._client is not None:
            return True
        try:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        """后端是否可用。"""
        return self._ensure_client() and bool(self.api_key)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """调用 OpenAI 兼容聊天补全 API。"""
        if not self.available:
            raise RuntimeError(
                "OpenAI 后端不可用：请安装 openai SDK 并设置 OPENAI_API_KEY"
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI 请求失败: {e}") from e

    def embedding(self, text: str) -> List[float]:
        """调用 OpenAI 嵌入 API。"""
        if not self.available:
            raise RuntimeError("OpenAI 后端不可用")

        try:
            response = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI Embedding 请求失败: {e}") from e


# ══════════════════════════════════════════════════════════════
# Mock 后端（用于测试）
# ══════════════════════════════════════════════════════════════
class MockBackend(LLMBackend):
    """模拟 LLM 后端，用于测试和无需 API 的场景。

    不调用任何真实 API，返回基于规则的模拟结果。
    """

    # 回应类型关键词
    _EXTEND_HINTS = ["对", "是的", "没错", "确实", "同意", "有道理"]
    _SHIFT_HINTS = ["说到", "其实", "不过", "另外", "但是"]
    _AVOID_HINTS = ["嗯", "好吧", "差不多", "看情况"]

    @property
    def available(self) -> bool:
        """Mock 后端始终可用。"""
        return True

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """模拟聊天补全。

        根据 user_prompt 内容返回模拟分类结果。
        如果 prompt 包含回应类型分类相关内容，返回分类结果。
        """
        # 检测是否是回应分类请求
        if "extend" in system_prompt and "shift" in system_prompt:
            return self._mock_response_classification(user_prompt)

        # 检测是否是核心观点提取请求
        if "核心观点" in system_prompt or "core point" in system_prompt.lower():
            return self._mock_core_point_extraction(user_prompt)

        # 检测是否是论据类型分类请求
        if "论据" in system_prompt or "evidence" in system_prompt.lower():
            return "personal_experience"

        # 默认返回
        return "mock_response"

    def _mock_response_classification(self, user_prompt: str) -> str:
        """模拟回应类型分类。"""
        lower = user_prompt.lower()
        for hint in self._EXTEND_HINTS:
            if hint in user_prompt:
                return "extend"
        for hint in self._SHIFT_HINTS:
            if hint in user_prompt:
                return "shift"
        for hint in self._AVOID_HINTS:
            if hint in user_prompt:
                return "avoid"
        return "extend"

    def _mock_core_point_extraction(self, user_prompt: str) -> str:
        """模拟核心观点提取。"""
        # 返回一个 JSON 格式的模拟结果
        return json.dumps([
            {"point": "模拟观点1", "novelty": 0.7},
            {"point": "模拟观点2", "novelty": 0.5},
        ], ensure_ascii=False)

    def embedding(self, text: str) -> List[float]:
        """返回随机嵌入向量。"""
        random.seed(hash(text) % (2**32))
        return [random.uniform(-1, 1) for _ in range(128)]


# ══════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════
def get_llm_backend(
    backend_name: str = "auto",
    **kwargs: Any,
) -> LLMBackend:
    """根据名称获取 LLM 后端实例。

    Args:
        backend_name: 后端名称，支持以下值:
            - 'auto': 自动选择可用的后端（优先 dashscope，其次 openai，最后 mock）
            - 'dashscope': 阿里云百炼后端
            - 'openai': OpenAI 兼容后端
            - 'mock': 模拟后端（始终可用，用于测试）
        **kwargs: 后端特定参数，将传递给后端构造函数。
            dashscope 后端参数: api_key, model, embedding_model
            openai 后端参数: api_key, base_url, model

    Returns:
        LLMBackend 实例。

    Raises:
        ValueError: 当 backend_name 不支持时。
    """
    backend_name = backend_name.lower().strip()

    if backend_name == "mock":
        return MockBackend()

    elif backend_name == "dashscope":
        return DashScopeBackend(**kwargs)

    elif backend_name == "openai":
        return OpenAICompatibleBackend(**kwargs)

    elif backend_name == "auto":
        # 按优先级尝试各后端
        candidates = [
            ("dashscope", lambda: DashScopeBackend(**kwargs)),
            ("openai", lambda: OpenAICompatibleBackend(**kwargs)),
        ]

        for name, factory in candidates:
            try:
                backend = factory()
                if backend.available:
                    print(f"  [LLM] 自动选择后端: {name}")
                    return backend
            except Exception:
                continue

        # 所有后端都不可用，返回 Mock
        print("  [LLM] 无可用 LLM 后端，使用 Mock（将降级为纯规则分析）")
        return MockBackend()

    else:
        raise ValueError(
            f"不支持的 LLM 后端: '{backend_name}'。"
            f"支持的后端: 'auto', 'dashscope', 'openai', 'mock'"
        )


# ══════════════════════════════════════════════════════════════
# 内容分析器
# ══════════════════════════════════════════════════════════════
class ContentAnalyzer:
    """内容分析器，使用 LLM 后端做内容层面的语义分析。

    封装了常用的内容分析方法，包括:
    - 回应类型分类（LLM 增强）
    - 核心观点提取
    - 论据类型分类
    - 信息新颖度评估

    Attributes:
        llm_backend: LLM 后端实例。
        config: 分析配置（prompt 模板等）。
    """

    # 默认 prompt 配置
    DEFAULT_CONFIG: Dict[str, Any] = {
        "response_classification": {
            "system_prompt": (
                "你是一个对话分析专家。你的任务是分析对话中回应者对提问的回应类型。\n"
                "回应类型有四种：\n"
                "- extend: 延展——接住对方内容并添加新信息或观点\n"
                "- shift: 转移——切换话题或未回应对方的内容\n"
                "- avoid: 回避——极短回应或敷衍\n"
                "- repeat: 重复——重复已有内容，无新增信息\n"
                "请只输出一个单词：extend、shift、avoid 或 repeat。"
            ),
            "user_template": (
                "嘉宾说：{question}\n\n"
                "主播回应：{response}\n\n"
                "请分类这个回应的类型。"
            ),
        },
        "core_point_extraction": {
            "system_prompt": (
                "你是一个内容分析专家。请从以下文本中提取核心观点。\n"
                "输出 JSON 数组，每个元素包含:\n"
                '- "point": 观点描述（一句话）\n'
                '- "novelty": 新颖度 [0-1]，1表示全新观点，0表示常识\n'
                '- "evidence_type": 论据类型，可选值: data/case/analogy/quote/personal_experience/none\n'
                "只输出 JSON，不要其他文字。"
            ),
            "user_template": "{text}",
        },
        "evidence_type_detection": {
            "system_prompt": (
                "你是一个内容分析专家。请判断以下文本主要使用的论据类型。\n"
                "可选值: data(数据)、case(案例)、analogy(类比)、"
                "quote(引用)、personal_experience(个人经验)、none(无明显论据)\n"
                "只输出一个词。"
            ),
            "user_template": "{text}",
        },
        "argument_structure_extraction": {
            "system_prompt": (
                "你是一个论证分析专家。请从以下文本中提取论证结构。\n"
                "分析三个要素：\n"
                "- claim: 主张（作者的核心观点）\n"
                "- evidence: 论据（支撑主张的事实、数据、案例）\n"
                "- conclusion: 结论（总结性判断）\n"
                "输出 JSON，格式：\n"
                '{"has_claim": true/false, "has_evidence": true/false, '
                '"has_conclusion": true/false, "completeness": 0.0-1.0}\n'
                "completeness: 三要素齐全=1.0，两个=0.67，一个=0.33，零个=0\n"
                "只输出 JSON，不要其他文字。"
            ),
            "user_template": "{text}",
        },
        "thinking_depth_classification": {
            "system_prompt": (
                "你是一个认知分析专家。请判断以下文本的思维深度层级。\n"
                "四个层级：\n"
                "- surface: 表面层——现象描述、事实陈述、个人经历、场景描述\n"
                "- analysis: 分析层——因果推导、对比分析、逻辑拆解、原因探究\n"
                "- synthesis: 综合层——框架构建、规律提炼、本质洞察、体系化思考\n"
                "- meta: 元认知层——自我反思、认知校准、方法论探讨\n"
                "请只输出一个单词：surface、analysis、synthesis 或 meta。"
            ),
            "user_template": "{text}",
        },
    }

    def __init__(
        self,
        llm_backend: Optional[LLMBackend] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化内容分析器。

        Args:
            llm_backend: LLM 后端实例。为 None 则自动选择。
            config: 分析配置。为 None 则使用默认配置。
        """
        self.llm_backend = llm_backend or get_llm_backend("auto")
        self.config = config or self.DEFAULT_CONFIG
        self._is_mock = isinstance(self.llm_backend, MockBackend)

    @property
    def is_mock(self) -> bool:
        """是否使用 Mock 后端（即 LLM 不可用）。"""
        return self._is_mock

    def classify_response(
        self,
        question_text: str,
        response_text: str,
    ) -> Dict[str, Any]:
        """使用 LLM 分类回应类型。

        Args:
            question_text: 提问文本。
            response_text: 回应文本。

        Returns:
            分类结果字典:
                response_type: extend/shift/avoid/repeat
                confidence: 置信度
        """
        cfg = self.config.get("response_classification", self.DEFAULT_CONFIG["response_classification"])

        try:
            user_prompt = cfg["user_template"].format(
                question=question_text[:500],
                response=response_text[:500],
            )
            result = self.llm_backend.chat(
                system_prompt=cfg["system_prompt"],
                user_prompt=user_prompt,
                temperature=0.1,
            )
            result = result.strip().lower()

            for rtype in ["extend", "shift", "avoid", "repeat"]:
                if rtype in result:
                    return {"response_type": rtype, "confidence": 0.80}

            return {"response_type": "shift", "confidence": 0.30}
        except Exception as e:
            return {"response_type": "shift", "confidence": 0.20, "error": str(e)}

    def extract_core_points(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """使用 LLM 提取核心观点。

        Args:
            text: 输入文本。

        Returns:
            核心观点列表，每个元素包含:
                point: 观点描述
                novelty: 新颖度 [0-1]
                evidence_type: 论据类型
        """
        if self._is_mock:
            # Mock 模式：使用规则提取
            from analyzer.dialogue_parser import extract_key_phrases
            phrases = extract_key_phrases(text, max_phrases=5)
            return [
                {
                    "point": p,
                    "novelty": 0.5,
                    "evidence_type": "none",
                }
                for p in phrases
            ]

        cfg = self.config.get("core_point_extraction", self.DEFAULT_CONFIG["core_point_extraction"])

        try:
            user_prompt = cfg["user_template"].format(text=text[:2000])
            result = self.llm_backend.chat(
                system_prompt=cfg["system_prompt"],
                user_prompt=user_prompt,
                temperature=0.2,
            )

            # 解析 JSON
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[-1].rsplit("```", 1)[0]

            points = json.loads(result)
            if isinstance(points, list):
                return points
            return []
        except Exception as e:
            print(f"  [ContentAnalyzer] 核心观点提取失败: {e}")
            return []

    def detect_evidence_type(self, text: str) -> str:
        """使用 LLM 检测论据类型。

        Args:
            text: 输入文本。

        Returns:
            论据类型: data/case/analogy/quote/personal_experience/none
        """
        if self._is_mock:
            # Mock 模式：规则检测
            if any(w in text for w in ["%", "亿", "万", "数据", "统计"]):
                return "data"
            if any(w in text for w in ["比如", "举个例子", "案例"]):
                return "case"
            if any(w in text for w in ["就像", "好比", "相当于"]):
                return "analogy"
            if any(w in text for w in ["我", "我自己", "我的"]):
                return "personal_experience"
            return "none"

        cfg = self.config.get("evidence_type_detection", self.DEFAULT_CONFIG["evidence_type_detection"])

        try:
            user_prompt = cfg["user_template"].format(text=text[:1000])
            result = self.llm_backend.chat(
                system_prompt=cfg["system_prompt"],
                user_prompt=user_prompt,
                temperature=0.1,
            )
            result = result.strip().lower()

            for etype in ["data", "case", "analogy", "quote", "personal_experience", "none"]:
                if etype in result:
                    return etype
            return "none"
        except Exception:
            return "none"

    def extract_argument_structure(self, text: str) -> Dict[str, Any]:
        """使用 LLM 提取论证结构（主张→论据→结论）。

        Args:
            text: 输入文本。

        Returns:
            论证结构字典:
                has_claim: 是否包含主张
                has_evidence: 是否包含论据
                has_conclusion: 是否包含结论
                completeness: 完整性 [0-1]
        """
        if self._is_mock:
            # Mock 模式：基于关键词的规则判断
            claim_markers = ["我觉得", "我认为", "我的判断", "本质上"]
            evidence_markers = ["比如", "数据", "案例", "实际上"]
            conclusion_markers = ["所以", "因此", "核心", "本质", "关键"]
            return {
                "has_claim": any(m in text for m in claim_markers),
                "has_evidence": any(m in text for m in evidence_markers),
                "has_conclusion": any(m in text for m in conclusion_markers),
                "completeness": safe_float(
                    sum([
                        any(m in text for m in claim_markers),
                        any(m in text for m in evidence_markers),
                        any(m in text for m in conclusion_markers),
                    ]) / 3.0
                ),
                "logic_connectors": {},
            }

        cfg = self.config.get(
            "argument_structure_extraction",
            self.DEFAULT_CONFIG["argument_structure_extraction"],
        )

        try:
            user_prompt = cfg["user_template"].format(text=text[:2000])
            result = self.llm_backend.chat(
                system_prompt=cfg["system_prompt"],
                user_prompt=user_prompt,
                temperature=0.1,
            )

            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[-1].rsplit("```", 1)[0]

            parsed = json.loads(result)
            return {
                "has_claim": bool(parsed.get("has_claim", False)),
                "has_evidence": bool(parsed.get("has_evidence", False)),
                "has_conclusion": bool(parsed.get("has_conclusion", False)),
                "completeness": parsed.get("completeness", 0.0),
                "logic_connectors": {},
            }
        except Exception as e:
            print(f"  [ContentAnalyzer] 论证结构提取失败: {e}")
            return {
                "has_claim": False,
                "has_evidence": False,
                "has_conclusion": False,
                "completeness": 0.0,
                "logic_connectors": {},
            }

    def classify_thinking_depth(self, text: str) -> tuple:
        """使用 LLM 分类思维深度层级。

        Args:
            text: 输入文本。

        Returns:
            元组 (depth_level, confidence):
                depth_level: surface/analysis/synthesis/meta
                confidence: 置信度 [0-1]
        """
        if self._is_mock:
            # Mock 模式：基于关键词的简单规则
            if any(m in text for m in ["本质", "底层", "框架", "第一性"]):
                return "synthesis", 0.65
            if any(m in text for m in ["因为", "所以", "为什么", "对比"]):
                return "analysis", 0.60
            return "surface", 0.55

        cfg = self.config.get(
            "thinking_depth_classification",
            self.DEFAULT_CONFIG["thinking_depth_classification"],
        )

        try:
            user_prompt = cfg["user_template"].format(text=text[:2000])
            result = self.llm_backend.chat(
                system_prompt=cfg["system_prompt"],
                user_prompt=user_prompt,
                temperature=0.1,
            )
            result = result.strip().lower()

            for level in ["surface", "analysis", "synthesis", "meta"]:
                if level in result:
                    return level, 0.80

            return "surface", 0.30
        except Exception as e:
            print(f"  [ContentAnalyzer] 思维深度分类失败: {e}")
            return "surface", 0.20
