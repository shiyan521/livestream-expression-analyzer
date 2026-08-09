"""
跨场次追踪与对比分析模块。

加载多场直播的分析结果，提取三层指标，生成跨场次对比表和趋势分析。

用法:
  from analyzer.cross_session import CrossSessionTracker

  tracker = CrossSessionTracker()
  tracker.add_session("0727_连麦", "test_cognitive_result.json")
  tracker.add_session("0805_连麦", "test_cognitive_0805_result.json")
  report = tracker.generate_comparison()
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from analyzer.utils import safe_float


# ══════════════════════════════════════════════════════════════
# 指标定义
# ══════════════════════════════════════════════════════════════

# 三层指标定义: (指标key, 显示名, 层级, 单位, 方向)
# 方向: "higher_better" 或 "lower_better"
METRIC_DEFINITIONS: List[Tuple[str, str, str, str, str]] = [
    # ── 内容层 ──
    ("catch_rate",            "对话接球率",     "内容层", "%", "higher_better"),
    ("information_increment", "信息增量密度",   "内容层", "%", "higher_better"),
    ("topic_coherence",       "话题连贯性",     "内容层", "分", "higher_better"),
    ("content_depth",         "内容深度",       "内容层", "分", "higher_better"),
    ("content_quality_score", "内容质量综合",   "内容层", "分", "higher_better"),
    # ── 认知层 ──
    ("argument_completeness", "论证完整性",     "认知层", "分", "higher_better"),
    ("complete_ratio",        "完整论证比例",   "认知层", "%", "higher_better"),
    ("logic_jump_ratio",      "逻辑跳跃率",     "认知层", "%", "lower_better"),
    ("thinking_depth",        "思维深度",       "认知层", "分", "higher_better"),
    ("avg_depth_score",       "平均深度评分",   "认知层", "/4", "higher_better"),
    ("high_level_ratio",      "高层级占比",     "认知层", "%", "higher_better"),
    ("surface_dominance",     "表面层占比",     "认知层", "%", "lower_better"),
    ("cognitive_resilience",  "认知韧性",       "认知层", "分", "higher_better"),
    ("cognitive_quality",     "认知质量综合",   "认知层", "分", "higher_better"),
    # ── 认知-内容关联 ──
    ("pearson_r",             "负荷-质量相关性", "关联",   "r", "higher_better"),
    ("overload_zones",        "过载区数",       "关联",   "段", "lower_better"),
    ("optimal_zones",         "最佳区数",       "关联",   "段", "higher_better"),
    ("degradation_pct",       "质量退化幅度",   "关联",   "%", "higher_better"),
]


# ══════════════════════════════════════════════════════════════
# 跨场次追踪器
# ══════════════════════════════════════════════════════════════
class CrossSessionTracker:
    """跨场次指标追踪器。

    加载多场直播分析结果，提取统一格式的指标，
    生成对比表和趋势分析。

    Attributes:
        sessions: 已加载的场次列表，每个包含 name, data, metrics
    """

    def __init__(self) -> None:
        self.sessions: List[Dict[str, Any]] = []

    def add_session(self, name: str, result_path: str) -> None:
        """加载一场直播的分析结果。

        Args:
            name: 场次名称（如 "0727_连麦"）。
            result_path: 分析结果 JSON 文件路径。
        """
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = self._extract_metrics(data)
        self.sessions.append({
            "name": name,
            "data": data,
            "metrics": metrics,
            "path": result_path,
        })
        print(f"  已加载: {name} ({len(metrics)} 个指标)")

    def add_session_data(self, name: str, data: Dict[str, Any]) -> None:
        """直接添加分析结果数据（不通过文件）。

        Args:
            name: 场次名称。
            data: 分析结果字典。
        """
        metrics = self._extract_metrics(data)
        self.sessions.append({
            "name": name,
            "data": data,
            "metrics": metrics,
            "path": None,
        })

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """从分析结果中提取统一格式的指标。

        Args:
            data: 完整分析结果字典。

        Returns:
            指标名→值的字典。
        """
        m: Dict[str, float] = {}
        content = data.get("content_analysis", {})
        cognitive = data.get("cognitive_analysis", {})
        content_summary = content.get("summary", {})
        cog_summary = cognitive.get("summary", {})

        # ── 内容层 ──
        catch = content.get("catch_rate_analysis", {})
        m["catch_rate"] = safe_float(catch.get("overall_catch_rate", 0)) * 100

        increment = content.get("information_density", {})
        m["information_increment"] = safe_float(
            increment.get("avg_increment_ratio", 0)
        ) * 100

        coherence = content.get("topic_coherence", {})
        m["topic_coherence"] = safe_float(coherence.get("avg_relevance", 0)) * 100

        depth = content.get("content_depth", {})
        m["content_depth"] = safe_float(depth.get("avg_novelty_score", 0)) * 100

        m["content_quality_score"] = safe_float(
            content_summary.get("content_quality_score", 0)
        )

        # ── 认知层 ──
        arg = cognitive.get("argument_chains", {})
        m["argument_completeness"] = safe_float(
            cog_summary.get("argument_score", 0)
        )
        m["complete_ratio"] = safe_float(arg.get("complete_ratio", 0)) * 100
        m["logic_jump_ratio"] = safe_float(arg.get("logic_jump_ratio", 0)) * 100

        td = cognitive.get("thinking_depth", {})
        m["thinking_depth"] = safe_float(cog_summary.get("thinking_depth_score", 0))
        m["avg_depth_score"] = safe_float(td.get("avg_depth_score", 0))
        m["high_level_ratio"] = safe_float(td.get("high_level_ratio", 0)) * 100
        m["surface_dominance"] = safe_float(td.get("surface_dominance", 0)) * 100

        m["cognitive_resilience"] = safe_float(
            cog_summary.get("cognitive_resilience_score", 0)
        )
        m["cognitive_quality"] = safe_float(
            cog_summary.get("cognitive_quality_score", 0)
        )

        # ── 认知-内容关联 ──
        corr = cognitive.get("cognitive_content_correlation", {})
        m["pearson_r"] = safe_float(
            corr.get("correlation", {}).get("pearson_r", 0)
        )
        m["overload_zones"] = float(
            len(corr.get("overload_zones", []))
        )
        m["optimal_zones"] = float(
            len(corr.get("optimal_zones", []))
        )
        deg = corr.get("quality_degradation", {})
        m["degradation_pct"] = safe_float(deg.get("degradation_pct", 0))

        return m

    def generate_comparison(self) -> Dict[str, Any]:
        """生成跨场次对比分析。

        Returns:
            对比分析结果:
                sessions: 场次名称列表
                metrics_table: 指标对比表（行=指标，列=场次）
                trend_analysis: 趋势分析（首场→末场的变化）
                session_details: 各场次详细信息
        """
        if len(self.sessions) < 2:
            return {
                "error": "至少需要 2 场才能对比",
                "sessions": [s["name"] for s in self.sessions],
            }

        session_names = [s["name"] for s in self.sessions]

        # 构建对比表
        metrics_table: List[Dict[str, Any]] = []
        for key, display, layer, unit, direction in METRIC_DEFINITIONS:
            row: Dict[str, Any] = {
                "metric": display,
                "key": key,
                "layer": layer,
                "unit": unit,
                "direction": direction,
                "values": [],
            }
            for s in self.sessions:
                row["values"].append(s["metrics"].get(key, 0))
            metrics_table.append(row)

        # 趋势分析：首场 vs 末场
        first = self.sessions[0]["metrics"]
        last = self.sessions[-1]["metrics"]
        trends: List[Dict[str, Any]] = []
        for key, display, layer, unit, direction in METRIC_DEFINITIONS:
            v1 = first.get(key, 0)
            v2 = last.get(key, 0)
            delta = v2 - v1
            if direction == "higher_better":
                status = "improving" if delta > 0 else ("declining" if delta < 0 else "stable")
            else:
                status = "improving" if delta < 0 else ("declining" if delta > 0 else "stable")
            trends.append({
                "metric": display,
                "key": key,
                "first": v1,
                "last": v2,
                "delta": delta,
                "status": status,
            })

        # 各场次详细信息
        session_details: List[Dict[str, Any]] = []
        for s in self.sessions:
            detail = {
                "name": s["name"],
                "metrics": s["metrics"],
            }
            # 提取时间段状态
            corr = s["data"].get("cognitive_analysis", {}).get(
                "cognitive_content_correlation", {}
            )
            zones = corr.get("zones", [])
            detail["zones"] = [
                {
                    "segment": z.get("segment", ""),
                    "load_level": z.get("load_level", ""),
                    "catch_rate": z.get("catch_rate", 0),
                    "information_increment": z.get("information_increment", 0),
                    "zone_type": z.get("zone_type", ""),
                }
                for z in zones
            ]
            # 提取思维深度分布
            td = s["data"].get("cognitive_analysis", {}).get("thinking_depth", {})
            dist = td.get("depth_distribution", {})
            detail["depth_distribution"] = {
                level: dist.get(level, {}).get("count", 0)
                for level in ["surface", "analysis", "synthesis", "meta"]
            }
            session_details.append(detail)

        return {
            "sessions": session_names,
            "metrics_table": metrics_table,
            "trend_analysis": trends,
            "session_details": session_details,
        }

    def generate_markdown_table(self, comparison: Optional[Dict[str, Any]] = None) -> str:
        """生成 Markdown 格式的对比表。

        Args:
            comparison: generate_comparison() 的输出。为 None 则自动生成。

        Returns:
            Markdown 表格字符串。
        """
        if comparison is None:
            comparison = self.generate_comparison()
        if "error" in comparison:
            return comparison["error"]

        sessions = comparison["sessions"]
        lines: List[str] = []

        # 按层级分组
        current_layer = ""
        for row in comparison["metrics_table"]:
            if row["layer"] != current_layer:
                current_layer = row["layer"]
                lines.append(f"\n### {current_layer}\n")
                header = "| 指标 | " + " | ".join(sessions) + " | 趋势 |"
                sep = "|------|" + "|".join(["------"] * len(sessions)) + "|------|"
                lines.append(header)
                lines.append(sep)

            vals = row["values"]
            # 格式化数值
            formatted = []
            for v in vals:
                if row["unit"] == "%":
                    formatted.append(f"{v:.1f}%")
                elif row["unit"] == "/4":
                    formatted.append(f"{v:.2f}")
                elif row["unit"] == "r":
                    formatted.append(f"{v:.3f}")
                else:
                    formatted.append(f"{v:.1f}")
                _ = formatted  # avoid lint

            # 趋势箭头
            if len(vals) >= 2:
                delta = vals[-1] - vals[0]
                if row["direction"] == "higher_better":
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                else:
                    arrow = "↓" if delta > 0 else ("↑" if delta < 0 else "→")
            else:
                arrow = "→"

            line = f"| {row['metric']} | " + " | ".join(formatted) + f" | {arrow} |"
            lines.append(line)

        return "\n".join(lines)

    def save_comparison(self, output_path: str, fmt: str = "json") -> None:
        """保存对比结果。

        Args:
            output_path: 输出文件路径。
            fmt: 格式，'json' 或 'markdown'。
        """
        comparison = self.generate_comparison()
        if fmt == "markdown":
            md = self.generate_markdown_table(comparison)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"  对比结果已保存: {output_path}")


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="跨场次指标对比")
    parser.add_argument(
        "--sessions", nargs="+", required=True,
        help="场次结果文件，格式: name=path 或直接 path",
    )
    parser.add_argument("--output", default="cross_session_comparison.json")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    tracker = CrossSessionTracker()
    for spec in args.sessions:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = os.path.basename(spec).replace(".json", "")
            path = spec
        tracker.add_session(name, path)

    tracker.save_comparison(args.output, args.format)
