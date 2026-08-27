# -*- coding: utf-8 -*-
"""Day4 配图生成脚本：LangGraph CP1-CP3 全链路

运行：python make_figures.py
输出：本文件同级 img/ 目录中的 SVG 技术图。

技术图全部由 Matplotlib 原生绘制，部分为纯 SVG 字符串拼接；
不依赖 Faro 生图，确保公式与结构精确可复现。
"""
from __future__ import annotations
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

COLORS = {
    "ink": "#1f2933",
    "muted": "#56616f",
    "line": "#cfd8e3",
    "blue": "#2b6cb0",
    "blue_soft": "#dcecff",
    "red": "#b31b1b",
    "red_soft": "#fbe4e4",
    "orange": "#d97706",
    "orange_soft": "#fff1d6",
    "green": "#2f855a",
    "green_soft": "#ddf4e6",
    "purple": "#6b46c1",
    "purple_soft": "#ede9fe",
    "paper": "#fafaf7",
}

def _save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / name}")

# ========== 1. State 三形态与多 Schema ==========
def fig_state_schemas():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"wspace": 0.22})
    items = [
        ("TypedDict\n（推荐）", ["访问：state[\"key\"]", "返回：dict 增量", "校验：轻量", "场景：教程/原型"], COLORS["blue"], "最简 · 零依赖"),
        ("dataclass", ["访问：state.key", "返回：新对象", "陷阱：双重累加", "场景：结构化数据"], COLORS["orange"], "对象 · 易踩坑"),
        ("Pydantic\nBaseModel", ["访问：state.key", "返回：dict/对象", "校验：严格", "场景：生产/接口"], COLORS["green"], "校验 · 生产级"),
    ]
    for ax, (title, lines, col, tag) in zip(axes, items):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        # card
        rect = plt.Rectangle((0.5, 0.5), 9, 9, fill=True, facecolor="white", edgecolor=col, linewidth=2, joinstyle="round")
        # matplotlib doesn't do round easily; approximate
        ax.add_patch(rect)
        ax.text(5, 8.6, title, ha="center", va="center", fontsize=13.5, fontweight="bold", color=col)
        ax.text(5, 7.7, tag, ha="center", va="center", fontsize=9.5, color=COLORS["muted"], style="italic")
        ax.plot([1.2, 8.8], [7.2, 7.2], color=COLORS["line"], lw=1.2)
        y = 6.2
        for line in lines:
            ax.text(1.5, y, f"• {line}", ha="left", va="center", fontsize=10.5, color=COLORS["ink"])
            y -= 1.0
        ax.text(5, 1.2, "StateGraph(state_schema=…)", ha="center", va="center", fontsize=9, color=COLORS["muted"], family="monospace")
    fig.text(0.5, 0.02, "三者在运行期等价，差异仅在“定义形式 × 访问语法 × 校验强度”；TypedDict 最贴合 LangGraph 的增量补丁模型。", ha="center", fontsize=10, color=COLORS["muted"])
    fig.suptitle("State 的三种声明形态（CP1-02）", fontsize=16, fontweight="bold", color=COLORS["ink"])
    _save(fig, "day4-state-schemas.svg")

# ========== 2. Reducer 归约语义 ==========
def fig_reducer():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    # header
    ax.text(5, 4.85, "Reducer：决定“多个节点对同一字段的写入如何合并”", ha="center", fontsize=11, fontweight="bold", color=COLORS["ink"])
    # columns
    cols = [
        (1.6, COLORS["blue"], "add（拼接）", "logs: Annotated[list, add]", "['a'] + ['b'] → ['a','b']", "并行安全", "logs 轨迹 / entries 汇总"),
        (5.0, COLORS["red"], "Overwrite（覆盖）", "logs: Overwrite(['b'])", "历史截断 → ['b']", "强制重置", "需要“重开一轮”的场景"),
        (8.4, COLORS["purple"], "add_messages", "messages: MessagesState", "按 id 去重与时序合并", "对话记忆", "多轮对话 / 工具循环"),
    ]
    for cx, col, title, code, example, feat, scene in cols:
        # box
        ax.add_patch(plt.Rectangle((cx-1.35, 0.6), 2.7, 3.6, fill=True, facecolor="white", edgecolor=col, linewidth=1.6))
        ax.text(cx, 3.85, title, ha="center", fontsize=10, fontweight="bold", color=col)
        ax.text(cx, 3.45, code, ha="center", fontsize=6.8, color=COLORS["muted"], family="monospace")
        ax.plot([cx-1.2, cx+1.2], [3.22, 3.22], color=COLORS["line"], lw=1)
        ax.text(cx, 2.75, example, ha="center", fontsize=7.2, color=COLORS["ink"])
        ax.text(cx, 2.25, f"特性：{feat}", ha="center", fontsize=7.5, color=col, fontweight="bold")
        ax.text(cx, 1.75, f"场景：{scene}", ha="center", fontsize=7, color=COLORS["muted"])
        ax.text(cx, 1.15, "✓" if "add" in title.lower() or "message" in title.lower() else "!", ha="center", fontsize=12, color=col)
    fig.text(0.5, 0.02, "无 Annotated → 默认覆盖语义（后写赢）；Annotated + reducer → 声明“该字段如何并发归约”。", ha="center", fontsize=8, color=COLORS["muted"])
    _save(fig, "day4-reducer-semantics.svg")

# ========== 3. 控制流家族拓扑 ==========
def fig_control_flows():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.18, hspace=0.32)
    configs = [
        ("扇出并行（CP2-01）\nSTART → {poem, joke}", [
            ("START", 0.5, 0.85), ("node_1\n(poem)", 0.2, 0.45), ("node_2\n(joke)", 0.8, 0.45)
        ], [("START","node_1"),("START","node_2")], "#2b6cb0", "两节点并行写不同字段，无竞争"),
        ("条件路由（CP2-02/03）\nadd_conditional_edges", [
            ("START", 0.5, 0.85), ("路由\ncontent_type", 0.5, 0.55), ("node_1", 0.2, 0.2), ("node_2", 0.8, 0.2)
        ], [("START","路由"),("路由","node_1"),("路由","node_2")], "#d97706", "Literal 分支 + path_map 解耦键与节点名"),
        ("动态分支 Send（CP2-06）\n输入决定并行度", [
            ("START", 0.5, 0.85), ("router\n→ Send×3", 0.5, 0.55), ("worker\n×3 并行", 0.5, 0.22)
        ], [("START","router"),("router","worker")], "#2f855a", "Send(\"worker\", {...}) 动态生成 N 个实例"),
        ("指令式跳转 Command（CP2-07）\ngoto + update", [
            ("START", 0.5, 0.85), ("router\n(Command)", 0.5, 0.55), ("poem_node", 0.2, 0.2), ("joke_node", 0.8, 0.2)
        ], [("START","router"),("router","poem_node"),("router","joke_node")], "#6b46c1", "节点内 return Command(goto=…, update={})"),
    ]
    centers = {
        "START": None, "node_1": None, "node_2": None, "路由": None, "router": None, "worker": None, "poem_node": None, "joke_node": None
    }
    for ax, (title, nodes, edges, col, note) in zip(axes.flat, configs):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color=col, pad=10)
        pos = {}
        for name, x, y in nodes:
            pos[name] = (x, y)
            # draw box
            if name == "START":
                ax.add_patch(plt.Circle((x, y), 0.07, fill=True, facecolor=col, edgecolor="white"))
                ax.text(x, y, "S", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
                ax.text(x, y-0.12, "START", ha="center", fontsize=8.5, color=COLORS["muted"])
            elif "路由" in name or "router" in name:
                # diamond
                diamond = plt.Polygon([(x, y+0.09),(x+0.09, y),(x, y-0.09),(x-0.09, y)], fill=True, facecolor=COLORS["orange_soft"], edgecolor=col, linewidth=1.2)
                ax.add_patch(diamond)
                ax.text(x, y, name.replace("\n", " "), ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")
            else:
                ax.add_patch(plt.Rectangle((x-0.13, y-0.07), 0.26, 0.14, fill=True, facecolor="white", edgecolor=col, linewidth=1.3))
                ax.text(x, y, name, ha="center", va="center", fontsize=9, color=COLORS["ink"])
        for a, b in edges:
            if a not in pos or b not in pos:
                continue
            xa, ya = pos[a]; xb, yb = pos[b]
            ax.annotate("", xy=(xb, yb+0.07 if yb< ya else yb), xytext=(xa, ya-0.07), arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.text(0.5, 0.04, note, ha="center", fontsize=8.5, color=COLORS["muted"], style="italic")
    fig.suptitle("控制流家族（CP2 前半）：从静态边到动态指令", fontsize=15, fontweight="bold", color=COLORS["ink"])
    _save(fig, "day4-control-flows.svg")

# ========== 4. 扇入 AND vs OR 时序图 ==========

if __name__ == '__main__':
    fig_state_schemas()
    fig_reducer()
    fig_control_flows()
    print('partial figures done')

# NOTE: 完整版 8 张图已生成，剩余 5 张图由独立脚本生成并置于 img/：
#   day4-fanin-and-or.svg, day4-checkpoint-timeline.svg,
#   day4-cache-ttl.svg, day4-mapreduce.svg, day4-loop-remaining.svg
# 详见 /tmp/gen_rest.py 与 /tmp/gen_more.py 的生成逻辑。

