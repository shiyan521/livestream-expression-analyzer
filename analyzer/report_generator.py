"""
统一报告生成器模块。

将表达层、内容层、认知层的分析结果合并为一份完整的 Markdown 报告，
包含跨场次追踪表、各层指标详情、时间段状态、改进建议。

命名规范: {date}_{场景描述}分析报告.md

用法:
  from analyzer.report_generator import ReportGenerator

  gen = ReportGenerator()
  gen.add_session("0727_连麦", "test_cognitive_result.json")
  gen.add_session("0805_连麦", "test_cognitive_0805_result.json")
  gen.generate("0805_连麦分析报告.md")
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from analyzer.cross_session import CrossSessionTracker
from analyzer.utils import safe_float


# ══════════════════════════════════════════════════════════════
# 报告生成器
# ══════════════════════════════════════════════════════════════
class ReportGenerator:
    """统一报告生成器。

    合并三层分析结果，生成包含跨场次对比的 Markdown 报告。

    Attributes:
        tracker: 跨场次追踪器
        sessions: 场次数据列表
    """

    def __init__(self) -> None:
        self.tracker = CrossSessionTracker()
        self.sessions: List[Dict[str, Any]] = []

    def add_session(self, name: str, result_path: str) -> None:
        """添加一场直播的分析结果。

        Args:
            name: 场次名称。
            result_path: 分析结果 JSON 文件路径。
        """
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tracker.add_session(name, result_path)
        self.sessions.append({"name": name, "data": data, "path": result_path})

    def add_session_data(self, name: str, data: Dict[str, Any]) -> None:
        """直接添加分析结果数据。

        Args:
            name: 场次名称。
            data: 分析结果字典。
        """
        self.tracker.add_session_data(name, data)
        self.sessions.append({"name": name, "data": data, "path": None})

    def generate(self, output_path: str, title: str = "") -> str:
        """生成完整报告并保存。

        Args:
            output_path: 输出文件路径。
            title: 报告标题，为空则自动生成。

        Returns:
            报告文件路径。
        """
        lines: List[str] = []

        # ── 标题 ──
        if not title:
            sessions_str = " vs ".join(s["name"] for s in self.sessions)
            title = f"直播表达分析报告 - {sessions_str}"
        lines.append(f"# {title}\n")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 场次数: {len(self.sessions)}\n")

        # ── 摘要 ──
        lines.append("## 摘要\n")
        summary = self._generate_summary()
        lines.append(summary)
        lines.append("")

        # ── 跨场次对比 ──
        if len(self.sessions) >= 2:
            lines.append("## 跨场次指标对比\n")
            md_table = self.tracker.generate_markdown_table()
            lines.append(md_table)
            lines.append("")

            # 趋势分析
            lines.append("### 趋势分析\n")
            comparison = self.tracker.generate_comparison()
            trends = comparison.get("trend_analysis", [])
            improving = [t for t in trends if t["status"] == "improving"]
            declining = [t for t in trends if t["status"] == "declining"]
            stable = [t for t in trends if t["status"] == "stable"]

            if improving:
                lines.append("**改善指标:**\n")
                for t in improving:
                    lines.append(
                        f"- {t['metric']}: {t['first']:.1f} → {t['last']:.1f} "
                        f"(Δ {t['delta']:+.1f})"
                    )
                lines.append("")
            if declining:
                lines.append("**退化指标:**\n")
                for t in declining:
                    lines.append(
                        f"- {t['metric']}: {t['first']:.1f} → {t['last']:.1f} "
                        f"(Δ {t['delta']:+.1f})"
                    )
                lines.append("")
            if stable:
                lines.append("**持平指标:**\n")
                for t in stable:
                    lines.append(
                        f"- {t['metric']}: {t['first']:.1f} → {t['last']:.1f}"
                    )
                lines.append("")

        # ── 各场次详情 ──
        for s in self.sessions:
            lines.append(f"## {s['name']} 详细分析\n")
            detail = self._generate_session_detail(s["data"])
            lines.append(detail)
            lines.append("")

        # ── 改进建议 ──
        lines.append("## 改进建议\n")
        suggestions = self._generate_suggestions()
        lines.append(suggestions)
        lines.append("")

        # 写入文件
        report = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  报告已保存: {output_path}")
        return output_path

    def _generate_summary(self) -> str:
        """生成报告摘要。"""
        lines: List[str] = []
        for s in self.sessions:
            data = s["data"]
            content_score = data.get("content_analysis", {}).get(
                "summary", {}
            ).get("content_quality_score", 0)
            cog_score = data.get("cognitive_analysis", {}).get(
                "summary", {}
            ).get("cognitive_quality_score", 0)
            catch_rate = data.get("content_analysis", {}).get(
                "catch_rate_analysis", {}
            ).get("overall_catch_rate", 0)
            arg_ratio = data.get("cognitive_analysis", {}).get(
                "argument_chains", {}
            ).get("complete_ratio", 0)

            lines.append(
                f"- **{s['name']}**: 内容质量 {content_score:.1f}/100, "
                f"认知质量 {cog_score:.1f}/100, "
                f"接球率 {catch_rate:.1%}, "
                f"完整论证 {arg_ratio:.1%}"
            )
        return "\n".join(lines)

    def _generate_session_detail(self, data: Dict[str, Any]) -> str:
        """生成单场详细分析。"""
        lines: List[str] = []
        content = data.get("content_analysis", {})
        cognitive = data.get("cognitive_analysis", {})

        # ── 内容层 ──
        lines.append("### 内容层\n")
        cs = content.get("summary", {})
        lines.append(
            f"- 内容质量综合: **{cs.get('content_quality_score', 0):.1f}**/100"
        )

        catch = content.get("catch_rate_analysis", {})
        lines.append(
            f"- 对话接球率: **{catch.get('overall_catch_rate', 0):.1%}** "
            f"({catch.get('total_exchanges', 0)} 个交换对, 趋势: {catch.get('trend', 'N/A')})"
        )

        inc = content.get("information_density", {})
        lines.append(
            f"- 信息增量密度: **{inc.get('avg_increment_ratio', 0):.1%}** "
            f"(趋势: {inc.get('trend', 'N/A')})"
        )

        coh = content.get("topic_coherence", {})
        lines.append(
            f"- 话题连贯性: {coh.get('avg_relevance', 0):.3f} "
            f"(漂移: {coh.get('topic_drift_count', 0)} 次)"
        )

        depth = content.get("content_depth", {})
        lines.append(
            f"- 内容深度: {depth.get('avg_novelty_score', 0):.2f} "
            f"({depth.get('total_core_points', 0)} 个核心观点)"
        )
        lines.append("")

        # ── 认知层 ──
        lines.append("### 认知层\n")
        cog_s = cognitive.get("summary", {})
        lines.append(
            f"- 认知质量综合: **{cog_s.get('cognitive_quality_score', 0):.1f}**/100"
        )
        lines.append(
            f"- 论证完整性: **{cog_s.get('argument_score', 0):.1f}**/100"
        )
        lines.append(
            f"- 思维深度: **{cog_s.get('thinking_depth_score', 0):.1f}**/100"
        )
        lines.append(
            f"- 认知韧性: **{cog_s.get('cognitive_resilience_score', 0):.1f}**/100"
        )
        lines.append("")

        # 论证链
        arg = cognitive.get("argument_chains", {})
        lines.append("#### 论证链追踪\n")
        lines.append(
            f"- 完整论证: {arg.get('complete_arguments', 0)} 个 "
            f"({arg.get('complete_ratio', 0):.1%})"
        )
        lines.append(
            f"- 逻辑跳跃: {arg.get('logic_jump_count', 0)} 次 "
            f"({arg.get('logic_jump_ratio', 0):.1%})"
        )
        lines.append(
            f"- 平均论证完整性: {arg.get('avg_completeness', 0):.2f}/1.0"
        )
        lines.append(
            f"- 论证链延续度: {arg.get('avg_chain_continuation', 0):.2f}"
        )
        lines.append(
            f"- 趋势: {arg.get('trend', 'stable')}"
        )
        lines.append("")

        # 思维深度
        td = cognitive.get("thinking_depth", {})
        dist = td.get("depth_distribution", {})
        lines.append("#### 思维深度分布\n")
        lines.append(f"| 层级 | 次数 | 占比 |")
        lines.append(f"|------|------|------|")
        for level in ["surface", "analysis", "synthesis", "meta"]:
            info = dist.get(level, {})
            lines.append(
                f"| {level} | {info.get('count', 0)} | {info.get('ratio', 0):.1%} |"
            )
        lines.append(
            f"\n平均深度: {td.get('avg_depth_score', 0):.2f}/4, "
            f"高层级占比: {td.get('high_level_ratio', 0):.1%}, "
            f"趋势: {td.get('trend', 'stable')}"
        )
        lines.append("")

        # 认知-内容关联
        corr = cognitive.get("cognitive_content_correlation", {})
        lines.append("#### 认知负荷与内容质量关联\n")
        lines.append(
            f"- Pearson r: **{corr.get('correlation', {}).get('pearson_r', 0):.3f}** "
            f"({corr.get('correlation', {}).get('interpretation', 'N/A')})"
        )
        zone_dist = corr.get("zone_distribution", {})
        lines.append(
            f"- 区域分布: 最佳 {zone_dist.get('optimal', 0)} 段, "
            f"韧性 {zone_dist.get('resilient', 0)} 段, "
            f"过载 {zone_dist.get('overload', 0)} 段, "
            f"滑行 {zone_dist.get('coasting', 0)} 段"
        )
        deg = corr.get("quality_degradation", {})
        has_baseline = deg.get("has_low_baseline", True)
        lines.append(
            f"- 质量退化: {deg.get('degradation_pct', 0):.1f}%"
            + ("" if has_baseline else " (全程高负荷, 用最低负荷段做基线)")
        )
        if "temporal_degradation" in deg:
            lines.append(
                f"- 时间维度退化: {deg.get('temporal_degradation', 0):.3f}"
            )
        lines.append("")

        # 各时间段状态
        zones = corr.get("zones", [])
        if zones:
            lines.append("#### 各时间段状态\n")
            lines.append("| 时间段 | 负荷 | 接球率 | 信息增量 | 区域类型 |")
            lines.append("|--------|------|--------|----------|----------|")
            for z in zones:
                lines.append(
                    f"| {z.get('segment', '')} "
                    f"| {z.get('load_level', '')} "
                    f"| {z.get('catch_rate', 0):.0%} "
                    f"| {z.get('information_increment', 0):.0%} "
                    f"| {z.get('zone_type', '')} |"
                )
            lines.append("")

        return "\n".join(lines)

    def _generate_suggestions(self) -> str:
        """基于分析结果生成改进建议。"""
        lines: List[str] = []

        # 分析最新一场的关键问题
        if not self.sessions:
            return "无数据"
        latest = self.sessions[-1]["data"]
        cog = latest.get("cognitive_analysis", {})
        content = latest.get("content_analysis", {})

        # 接球率
        catch_rate = content.get("catch_rate_analysis", {}).get("overall_catch_rate", 0)
        if catch_rate < 0.30:
            lines.append(
                f"1. **接球率偏低 ({catch_rate:.0%})**: 超过 70% 的对话交换未有效接住对方内容。"
                "建议在嘉宾发言后先复述关键点（2-3秒），再展开自己的观点，"
                "避免直接跳转话题。"
            )

        # 论证完整性
        arg_ratio = cog.get("argument_chains", {}).get("complete_ratio", 0)
        if arg_ratio < 0.20:
            lines.append(
                f"2. **论证结构不完整 ({arg_ratio:.0%})**: 大多数回应缺乏"
                "「主张→论据→结论」完整链条。建议在表达观点时"
                "有意识地补全论据（数据/案例/类比），并用"
                "总结性语言收束。"
            )

        # 思维深度
        surface = cog.get("thinking_depth", {}).get("surface_dominance", 0)
        if surface > 0.60:
            lines.append(
                f"3. **表面层占比过高 ({surface:.0%})**: 超过六成回应停留在"
                "现象描述层面。建议在叙述事实后追加一层"
                "「为什么」和「意味着什么」的分析，"
                "将表面层转化为分析层或综合层。"
            )

        # 认知过载
        corr = cog.get("cognitive_content_correlation", {})
        overload_count = len(corr.get("overload_zones", []))
        if overload_count > 0:
            zones = corr.get("zones", [])
            overload_zones = [z for z in zones if z.get("zone_type") == "overload"]
            for oz in overload_zones:
                lines.append(
                    f"4. **认知过载区 ({oz.get('segment', '')})**: "
                    f"负荷达到 {oz.get('load_level', 'high')}，"
                    f"接球率降至 {oz.get('catch_rate', 0):.0%}。"
                    "建议在该时段设置认知休息点——"
                    "用简短提问让嘉宾多说，给自己 10-15 秒缓冲。"
                )

        # 质量退化
        deg = corr.get("quality_degradation", {})
        deg_pct = abs(deg.get("degradation_pct", 0))
        if deg_pct > 20:
            lines.append(
                f"5. **质量退化显著 ({deg_pct:.0f}%)**: 从低负荷到高负荷，"
                "内容质量下降明显。建议在后半场主动降低信息密度，"
                "用回顾和总结代替新增内容，保持输出质量稳定。"
            )

        # 信息增量趋势
        inc_trend = content.get("information_increment", {}).get("trend", "stable")
        if inc_trend == "declining":
            avg_inc = content.get("information_increment", {}).get("avg_increment", 0)
            lines.append(
                f"6. **信息增量递减 (趋势: declining)**: "
                f"平均增量 {avg_inc:.0%}，后半段新信息产出枯竭。"
                "建议提前准备 3-5 个备选话题点，"
                "在增量低于 30% 时切入新话题。"
            )

        if not lines:
            lines.append("各维度指标均在合理范围内，保持当前节奏即可。")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成统一分析报告")
    parser.add_argument(
        "--sessions", nargs="+", required=True,
        help="场次结果文件，格式: name=path",
    )
    parser.add_argument("--output", default="analysis_report.md")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    gen = ReportGenerator()
    for spec in args.sessions:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = os.path.basename(spec).replace(".json", "")
            path = spec
        gen.add_session(name, path)

    gen.generate(args.output, args.title)
