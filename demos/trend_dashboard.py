"""🔮 設計趨勢儀表板 — 從 hhh 17,274 案例 × 15 年資料,看設計風格演變。

四大維度:
  📈 風格 popularity 演變 (line chart)
  🗺️ 地區 × 風格 分布 (heatmap)
  📊 每年案例量趨勢 (bar chart)
  🏆 各年 Top 5 風格 leaderboard

Claude 看圖表 + 統計數字,撰寫年度趨勢報告。
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
from PIL import Image

# Mac 中文字型
for _cjk in ["PingFang TC", "Heiti TC", "STHeiti", "Apple LiGothic", "Microsoft JhengHei"]:
    try:
        _fm.findfont(_cjk, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cjk]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_utils import connect

_DATA = None  # cache

# 風格配色 (給 chart 用)
STYLE_COLORS = {
    "現代風": "#3b82f6", "混搭風": "#ec4899", "休閒多元": "#10b981",
    "北歐風": "#f59e0b", "新古典": "#8b5cf6", "美式風": "#06b6d4",
    "奢華風": "#d4af37", "工業風": "#475569", "鄉村風": "#a16207",
    "古典風": "#9333ea", "其他": "#94a3b8", "日式禪風": "#84a98c",
    "東方風": "#dc2626", "日式風": "#16a34a", "前衛風": "#f97316",
}


def _load_data():
    """一次性撈所有案例 (style + year + location)。"""
    global _DATA
    if _DATA is not None:
        return _DATA
    print("[trend_dashboard] loading data from xoops _hcase ...", flush=True)
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT YEAR(creat_time) AS year, style, location, hdesigner_id
            FROM _hcase
            WHERE onoff = 1 AND creat_time IS NOT NULL AND YEAR(creat_time) >= 2012
            """
        )
        rows = cur.fetchall()
    conn.close()
    _DATA = rows
    print(f"[trend_dashboard] loaded {len(rows)} cases", flush=True)
    return _DATA


def _style_to_main(style: str) -> str:
    """把 '現代風,別墅' 這種多 tag 取第一個。"""
    if not style:
        return "未分類"
    return style.split(",")[0].strip() or "未分類"


# ===== Chart renderers =====
def _chart_style_trend(years_range: tuple[int, int], top_n: int = 6) -> Image.Image:
    """各風格 popularity 演變 line chart。"""
    data = _load_data()
    y_min, y_max = years_range
    # 統計 (year, style) → count
    counts: dict[int, dict[str, int]] = {}
    style_totals: dict[str, int] = {}
    for r in data:
        y, s = r["year"], _style_to_main(r["style"])
        if y is None or y < y_min or y > y_max:
            continue
        counts.setdefault(y, {}).setdefault(s, 0)
        counts[y][s] += 1
        style_totals[s] = style_totals.get(s, 0) + 1

    top_styles = sorted(style_totals.items(), key=lambda x: -x[1])[:top_n]
    top_keys = [s for s, _ in top_styles]
    years = sorted(counts.keys())

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=110)
    fig.patch.set_facecolor("white")
    for style in top_keys:
        ys = [counts[y].get(style, 0) for y in years]
        color = STYLE_COLORS.get(style, "#94a3b8")
        ax.plot(years, ys, marker="o", linewidth=2.5, markersize=6, label=style, color=color)
    ax.set_xlabel("年份", fontsize=11, color="#475569")
    ax.set_ylabel("案例數", fontsize=11, color="#475569")
    ax.set_title(f"Top {top_n} 風格演變  ({y_min}–{y_max})", fontsize=14, fontweight="700", pad=14, color="#0f172a")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(years)
    ax.tick_params(colors="#64748b")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _chart_yearly_volume(years_range: tuple[int, int]) -> Image.Image:
    data = _load_data()
    y_min, y_max = years_range
    counts: dict[int, int] = {}
    for r in data:
        y = r["year"]
        if y is None or y < y_min or y > y_max:
            continue
        counts[y] = counts.get(y, 0) + 1
    years = sorted(counts.keys())
    vals = [counts[y] for y in years]

    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=110)
    fig.patch.set_facecolor("white")
    bars = ax.bar(years, vals, color="#3b82f6", edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, f"{v:,}",
                ha="center", fontsize=9, color="#1e293b", fontweight="500")
    ax.set_xlabel("年份", fontsize=11, color="#475569")
    ax.set_ylabel("案例數", fontsize=11, color="#475569")
    ax.set_title(f"每年新增案例量  ({y_min}–{y_max})", fontsize=14, fontweight="700", pad=12, color="#0f172a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#64748b")
    ax.set_xticks(years)
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _chart_location_style(years_range: tuple[int, int], top_locations: int = 8, top_styles: int = 6) -> Image.Image:
    """地區 × 風格 heatmap (案例佔比)。"""
    data = _load_data()
    y_min, y_max = years_range
    loc_style: dict[str, dict[str, int]] = {}
    loc_totals: dict[str, int] = {}
    style_totals: dict[str, int] = {}
    for r in data:
        y, s, loc = r["year"], _style_to_main(r["style"]), (r["location"] or "").strip()
        if y is None or y < y_min or y > y_max or not loc:
            continue
        loc_style.setdefault(loc, {}).setdefault(s, 0)
        loc_style[loc][s] += 1
        loc_totals[loc] = loc_totals.get(loc, 0) + 1
        style_totals[s] = style_totals.get(s, 0) + 1

    top_loc = [k for k, _ in sorted(loc_totals.items(), key=lambda x: -x[1])[:top_locations]]
    top_sty = [k for k, _ in sorted(style_totals.items(), key=lambda x: -x[1])[:top_styles]]

    # 矩陣:row=loc, col=style, value=該地區該風格的「相對佔比」(正規化過)
    M = np.zeros((len(top_loc), len(top_sty)))
    for i, loc in enumerate(top_loc):
        loc_sum = loc_totals[loc]
        for j, sty in enumerate(top_sty):
            cnt = loc_style.get(loc, {}).get(sty, 0)
            M[i, j] = (cnt / loc_sum * 100) if loc_sum else 0

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=110)
    fig.patch.set_facecolor("white")
    im = ax.imshow(M, cmap="RdPu", aspect="auto")
    ax.set_xticks(range(len(top_sty)))
    ax.set_xticklabels(top_sty, fontsize=10, color="#475569")
    ax.set_yticks(range(len(top_loc)))
    ax.set_yticklabels(top_loc, fontsize=10, color="#475569")
    # 標數字
    for i in range(len(top_loc)):
        for j in range(len(top_sty)):
            val = M[i, j]
            color = "white" if val > 30 else "#0f172a"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, color=color, fontweight="600")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("案例佔比 %", color="#475569", fontsize=10)
    cb.ax.tick_params(colors="#64748b")
    ax.set_title(f"地區 × 風格分布  ({y_min}–{y_max})", fontsize=14, fontweight="700", pad=14, color="#0f172a")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _chart_yearly_top5(years_range: tuple[int, int]) -> Image.Image:
    """各年 Top 5 風格 horizontal bars per year."""
    data = _load_data()
    y_min, y_max = years_range
    # year → style → count
    by_year: dict[int, dict[str, int]] = {}
    for r in data:
        y, s = r["year"], _style_to_main(r["style"])
        if y is None or y < y_min or y > y_max:
            continue
        by_year.setdefault(y, {}).setdefault(s, 0)
        by_year[y][s] += 1

    years = sorted(by_year.keys())[-6:]  # 取最近 6 年方便看
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), dpi=110)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"近 {len(years)} 年 Top 5 風格", fontsize=14, fontweight="700", color="#0f172a", y=1.01)

    for idx, year in enumerate(years):
        ax = axes[idx // 3, idx % 3]
        top5 = sorted(by_year[year].items(), key=lambda x: -x[1])[:5]
        names = [t[0] for t in top5][::-1]
        vals = [t[1] for t in top5][::-1]
        colors = [STYLE_COLORS.get(n, "#94a3b8") for n in names]
        ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9, color="#475569")
        ax.set_title(f"{year}", fontsize=12, fontweight="600", color="#0f172a")
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, str(v), va="center", fontsize=9, color="#1e293b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#64748b")
        ax.set_xlim(0, max(vals) * 1.18)

    # 隱藏沒用到的 subplot
    for idx in range(len(years), 6):
        axes[idx // 3, idx % 3].axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ===== Claude 寫報告 =====
def _stats_summary(years_range: tuple[int, int]) -> dict:
    """整理一份 stats 給 Claude 參考。"""
    data = _load_data()
    y_min, y_max = years_range
    yearly: dict[int, dict[str, int]] = {}
    loc_count: dict[str, int] = {}
    total = 0
    designer_count = set()
    for r in data:
        y = r["year"]
        if y is None or y < y_min or y > y_max:
            continue
        s = _style_to_main(r["style"])
        loc = (r["location"] or "").strip()
        yearly.setdefault(y, {}).setdefault(s, 0)
        yearly[y][s] += 1
        if loc:
            loc_count[loc] = loc_count.get(loc, 0) + 1
        if r.get("hdesigner_id"):
            designer_count.add(r["hdesigner_id"])
        total += 1

    years = sorted(yearly.keys())
    # 每風格 5 年前 vs 近 1 年 變化
    if len(years) >= 2:
        recent = yearly[years[-1]]
        past = yearly[years[0]]
        all_styles = set(recent.keys()) | set(past.keys())
        changes = []
        for s in all_styles:
            past_v = past.get(s, 0)
            recent_v = recent.get(s, 0)
            change_pct = (recent_v - past_v) / max(past_v, 1) * 100 if past_v > 0 else (100 if recent_v else 0)
            changes.append({"style": s, "past": past_v, "recent": recent_v, "pct": change_pct})
        changes.sort(key=lambda x: -x["pct"])
    else:
        changes = []

    return {
        "year_range": f"{y_min}-{y_max}",
        "total_cases": total,
        "active_designers": len(designer_count),
        "top_styles_recent": sorted(yearly.get(years[-1], {}).items(), key=lambda x: -x[1])[:8],
        "rising_styles": changes[:5],
        "declining_styles": changes[-5:],
        "top_locations": sorted(loc_count.items(), key=lambda x: -x[1])[:8],
    }


def _claude_report(stats: dict, years_range: tuple[int, int]) -> str:
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "_⚠ Claude key 未設定,跳過報告生成_"
    client = Anthropic(api_key=key)
    prompt = f"""你是 hhh 幸福空間的數據分析師,要寫一份 **2026 室內設計趨勢報告**。
我給你以下統計資料,請你寫一份 350-500 字的趨勢報告,給設計師、媒體、屋主三方都看得懂。

統計資料 (年份範圍 {stats['year_range']}):
- 總案例數: {stats['total_cases']:,}
- 活躍設計師: {stats['active_designers']} 位
- 最近一年 Top 8 風格: {stats['top_styles_recent']}
- 5 年內漲幅最大 (rising): {stats['rising_styles']}
- 5 年內衰退最大 (declining): {stats['declining_styles']}
- Top 8 地區: {stats['top_locations']}

報告請包含:
1. **整體市場觀察** (1 段,2-3 句)
2. **崛起風格** (誰漲、漲多少、可能原因)
3. **衰退風格** (誰跌、原因推測)
4. **地區特性** (北中南差異)
5. **給設計師的建議**:接下來該朝什麼方向發展

格式:markdown,可用粗體、項目符號。繁體中文,專業但易讀。
"""
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ===== Handler =====
def analyze(year_start: int, year_end: int):
    print(f"[trend_dashboard] analyze {year_start}-{year_end}", flush=True)
    if year_end < year_start:
        year_start, year_end = year_end, year_start
    yr = (int(year_start), int(year_end))

    # 4 個 chart 同時生
    chart1 = _chart_style_trend(yr)
    chart2 = _chart_yearly_volume(yr)
    chart3 = _chart_location_style(yr)
    chart4 = _chart_yearly_top5(yr)

    # 統計 + Claude 報告
    stats = _stats_summary(yr)
    try:
        report = _claude_report(stats, yr)
    except Exception as e:
        report = f"_❌ 報告生成失敗:{e}_"

    summary_md = f"""
**📊 統計摘要**
- 時間範圍:{stats['year_range']}
- 涵蓋案例:**{stats['total_cases']:,}** 筆
- 活躍設計師:**{stats['active_designers']}** 位
"""
    return chart1, chart2, chart3, chart4, summary_md, report


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🔮",
            title="設計趨勢儀表板",
            subtitle="從 hhh 17,274 個案例 × 15 年資料,看設計風格如何演變 — 媒體級年度趨勢報告自動產生",
            tools=[
                ("xoops _hcase DB", "全量 17k+ 案例 (含 style/year/location)"),
                ("Claude Sonnet 4.6", "看統計數字撰寫趨勢報告"),
                ("matplotlib", "4 維度視覺化 (趨勢線 / 量化 / 熱度圖 / Top 5)"),
            ],
            cost="$0.05",
            cost_detail="只用 Claude 寫一次報告",
            time="10-15 秒",
            time_detail="圖表純本機算,Claude 寫報告 ~10s",
            badges=["數據洞察", "媒體素材", "B2B"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:選定年份範圍,系統用 hhh 17k 案例的真實資料畫出風格演變圖、地區熱度、Top 5 排行,Claude 看完寫一份「年度趨勢報告」。<br/>📌 適合:媒體 PR 素材、設計師方向建議、屋主了解市場動態。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                year_start = gr.Slider(2012, 2026, value=2018, step=1, label="起始年")
                year_end = gr.Slider(2012, 2026, value=2026, step=1, label="結束年")
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🔮 生成趨勢報告 (~15 秒)", variant="primary", scale=2)
                summary_md = gr.Markdown()
            with gr.Column(scale=3, elem_classes=["demo-output-pane"]):
                with gr.Row():
                    chart_trend = gr.Image(label="風格 popularity 演變", type="pil", height=320, interactive=False)
                    chart_volume = gr.Image(label="每年案例量", type="pil", height=320, interactive=False)
                with gr.Row():
                    chart_location = gr.Image(label="地區 × 風格分布", type="pil", height=320, interactive=False)
                    chart_top5 = gr.Image(label="近 6 年 Top 5 風格", type="pil", height=320, interactive=False)
                gr.Markdown("### 📄 AI 趨勢報告", elem_classes=["demo-section-title"])
                report_md = gr.Markdown()

        btn.click(
            analyze,
            inputs=[year_start, year_end],
            outputs=[chart_trend, chart_volume, chart_location, chart_top5, summary_md, report_md],
        )

    return demo
