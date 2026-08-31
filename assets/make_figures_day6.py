# -*- coding: utf-8 -*-
"""Day6 配图生成：matplotlib 替代 Mermaid，确保 GitHub Pages 可渲染。

生成：
  - day6-chain-matplotlib.png    五 Notebook 实验链
  - day6-unified-graph-matplotlib.png  猫主题并行流
  - day6-execution-order-matplotlib.png 推荐执行顺序

另有 faro 封面：
  - day6-cover-faro.png          gpt-image-2 封面（faroapi.com/v1）
  - day6-parallel-ambient.png    氛围背景（faroapi.com/v1）

运行：
  conda run -n LangChain python 08_MyNote/Day6/assets/img/make_figures.py
"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = pathlib.Path(r"D:\大学相关\03_个人成长与记录\LLM学习体系\08_MyNote\Day6\assets\img")
out.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

C = {"ink":"#1e2940","muted":"#6b7a90","line":"#d6deea","blue":"#2563eb","blue2":"#1e40af","amber":"#d97706","red":"#dc2626","green":"#0f8a5f","purple":"#7c3aed","bg":"#f8fafc"}

def save(fig, name):
    fig.savefig(out / name, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name} -> {out / name}")

# 1) 故障处理实验链（替代 flowchart TD）
def fig_chain():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0,13); ax.set_ylim(0,3.2); ax.axis("off")
    fig.patch.set_facecolor("white")
    steps = [
        ("CP3-04\n制造错误", C["red"], "#fee2e2"),
        ("START→\nchange_topic", C["blue"], "#dbeafe"),
        ("并行超步\npoem ✓ / joke ✗", C["amber"], "#fef3c7"),
        ("CP3-05\n查找错误", C["blue"], "#dbeafe"),
        ("CP3-06\n去掉故障\ninvoke None", C["green"], "#dcfce7"),
        ("CP3-07\nReplay\ncheckpoint_id", C["purple"], "#ede9fe"),
        ("CP3-08\nFork\nupdate_state", C["blue2"], "#e0e7ff"),
    ]
    x0, w, h, gap = 0.35, 1.55, 1.9, 0.28
    for i, (label, col, bg) in enumerate(steps):
        x = x0 + i*(w+gap)
        y = 0.7
        # 区分首尾形状
        rect = plt.Rectangle((x,y), w, h, fc=bg, ec=col, lw=1.8, joinstyle="round")
        # matplotlib rectangle with rounded corners via FancyBbox
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=bg, ec=col, lw=1.8)
        ax.add_patch(box)
        ax.text(x+w/2, y+h/2+0.22, label.split("\n")[0], ha="center", va="center", fontsize=9.8, weight="bold", color=col)
        rest = "\n".join(label.split("\n")[1:])
        if rest:
            ax.text(x+w/2, y+h/2-0.42, rest, ha="center", va="center", fontsize=7.8, color=C["ink"], linespacing=1.2)
        if i < len(steps)-1:
            ax.annotate("", xy=(x+w+gap-0.06, y+h/2), xytext=(x+w+0.06, y+h/2),
                        arrowprops=dict(arrowstyle="->", color=C["line"], lw=1.6, shrinkA=0, shrinkB=0))
            # 合并箭头特殊处理：E->F 在原图是两条汇入一条，这里用分叉点表示
        ax.text(6.5, 0.15, "一条连续的故障注入 → 定位 → 修复 → 重放 → 分支 实验链", ha="center", fontsize=8.8, color=C["muted"])
    ax.set_title("五个 Notebook 的连续实验链（替代 flowchart TD）", fontsize=12, weight="bold", color=C["ink"], pad=14)
    save(fig, "day6-chain-matplotlib.png")

# 2) 统一图结构（替代 flowchart LR 的猫主题图）
def fig_graph():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0,11); ax.set_ylim(0,3.6); ax.axis("off")
    fig.patch.set_facecolor("white")
    # positions
    pos = {
        "START": (1.1, 1.8),
        "change": (3.2, 1.8),
        "poem": (5.6, 2.85),
        "joke": (5.6, 0.75),
        "output": (8.4, 1.8),
        "END": (10.0, 1.8),
    }
    def box(label, sub, xy, col, bg):
        x,y = xy
        from matplotlib.patches import FancyBboxPatch
        if label in ("START","END"):
            c = plt.Circle((x,y), 0.42, fc=bg, ec=col, lw=1.8)
            ax.add_patch(c)
            ax.text(x,y, label, ha="center", va="center", fontsize=9, weight="bold", color=col)
            if sub: ax.text(x,y-0.62, sub, ha="center", fontsize=7, color=C["muted"])
        else:
            b = FancyBboxPatch((x-0.82,y-0.42), 1.64, 0.84, boxstyle="round,pad=0.02,rounding_size=0.16", fc=bg, ec=col, lw=1.6)
            ax.add_patch(b)
            ax.text(x,y+0.12, label, ha="center", fontsize=9, weight="bold", color=C["ink"])
            ax.text(x,y-0.22, sub, ha="center", fontsize=7, color=C["muted"])
    # draw edges
    def arrow(a,b, col=C["line"], style="-"):
        xa,ya = pos[a]; xb,yb = pos[b]
        # offset for circle
        import math
        if a in ("START","END"): xa,ya = xa+0.42 if xb>xa else xa-0.42, ya
        if b in ("START","END"): xb,yb = xb-0.42 if xb>xa else xb+0.42, yb
        # for boxes, adjust x
        if a not in ("START","END"): xa = xa+0.82 if xb>xa else xa-0.82
        if b not in ("START","END"):
            if abs(yb-ya)>0.3:
                # diagonal
                pass
            else:
                xb = xb-0.82 if xb>xa else xb+0.82
        ax.annotate("", xy=(xb,yb), xytext=(xa,ya), arrowprops=dict(arrowstyle="->", color=col, lw=1.7, connectionstyle="arc3,rad=0.02"))
    box("START","", pos["START"], C["blue"], "#dbeafe")
    box("change","node_change_topic\n猫→猫:子主题", pos["change"], C["blue"], "#dbeafe")
    box("poem","node_poem\n七言绝句", pos["poem"], C["green"], "#dcfce7")
    box("joke","node_joke\n笑话/故障注入", pos["joke"], C["red"], "#fee2e2")
    box("output","node_output\n汇总", pos["output"], C["purple"], "#ede9fe")
    box("END","", pos["END"], C["blue2"], "#e0e7ff")
    arrow("START","change", C["blue"])
    arrow("change","poem", C["line"])
    arrow("change","joke", C["line"])
    arrow("poem","output", C["green"])
    arrow("joke","output", C["red"])
    arrow("output","END", C["purple"])
    # fork annotation
    ax.text(4.35, 2.05, "扇出", ha="center", fontsize=7.5, color=C["muted"], style="italic")
    ax.text(6.95, 2.05, "扇入", ha="center", fontsize=7.5, color=C["muted"], style="italic")
    ax.text(5.5, 0.18, "扇出—并行—扇入：poem 与 joke 同一超步，output 需两者都完成才可执行", ha="center", fontsize=8.2, color=C["muted"])
    ax.set_title("统一图结构：猫主题并行流（替代 flowchart LR）", fontsize=12, weight="bold", color=C["ink"], pad=12)
    save(fig, "day6-unified-graph-matplotlib.png")

# 3) 执行顺序（替代最后的 flowchart LR）
def fig_order():
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(0,11); ax.set_ylim(0,2.2); ax.axis("off")
    fig.patch.set_facecolor("white")
    nodes = ["CP3-01","CP3-02","CP3-03","CP3-04","CP3-05","CP3-06","CP3-07","CP3-08"]
    colors = [C["muted"],C["muted"],C["muted"],C["red"],C["amber"],C["green"],C["purple"],C["blue2"]]
    bgs = ["#f1f5f9","#f1f5f9","#f1f5f9","#fee2e2","#fef3c7","#dcfce7","#ede9fe","#e0e7ff"]
    for i,(n,col,bg) in enumerate(zip(nodes, colors, bgs)):
        x = 0.4 + i*1.32
        from matplotlib.patches import FancyBboxPatch
        b = FancyBboxPatch((x,0.65), 1.08, 0.72, boxstyle="round,pad=0.02,rounding_size=0.14", fc=bg, ec=col, lw=1.5)
        ax.add_patch(b)
        ax.text(x+0.54, 1.01, n, ha="center", fontsize=8.8, weight="bold", color=col)
        if i < len(nodes)-1:
            ax.annotate("", xy=(x+1.32-0.04, 1.01), xytext=(x+1.08+0.04, 1.01), arrowprops=dict(arrowstyle="->", color=C["line"], lw=1.4))
    ax.text(5.5, 0.28, "CP3-04~06 共用 chapter03-05；CP3-07 用 chapter03-08；CP3-08 用内存分支", ha="center", fontsize=7.8, color=C["muted"])
    ax.set_title("推荐执行顺序", fontsize=11, weight="bold", color=C["ink"], pad=10)
    save(fig, "day6-execution-order-matplotlib.png")

fig_chain(); fig_graph(); fig_order()
print("all matplotlib done")
