"""💸 AI 預算焦慮治療師 — 把抽象預算變成「能做到 vs 要妥協」的可視化決策表。

流程:
  1. 屋主填:坪數 / 房數 / 預算 / 風格偏好 / 必備功能
  2. Claude 分析該預算在台灣市場能達成多少 (%)
  3. 列出必要妥協項與省下金額
  4. matplotlib 視覺化預算分配
  5. 從 57k case cache 推薦預算內案例
"""

from __future__ import annotations

import io
import json
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
from PIL import Image

for _cjk in ["PingFang TC", "Heiti TC", "STHeiti", "Apple LiGothic", "Microsoft JhengHei"]:
    try:
        _fm.findfont(_cjk, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cjk]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


_CLAUDE = None


def _claude():
    global _CLAUDE
    if _CLAUDE is not None:
        return _CLAUDE
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-api03-xxx"):
        raise RuntimeError("ANTHROPIC_API_KEY 未設定")
    _CLAUDE = Anthropic(api_key=key)
    return _CLAUDE


STYLE_OPTIONS = ["北歐風", "現代簡約", "輕奢風", "日式禪風", "工業風", "古典歐式", "混搭風", "鄉村風"]
HAS_KIDS_OPTIONS = ["無", "1 小孩", "2 小孩", "3+ 小孩"]
MUST_HAVE_OPTIONS = ["中島廚房", "更衣室", "書房", "客房", "玄關櫃", "陽台/洗衣間", "電視牆設計", "間接照明"]


SYSTEM_PROMPT = """你是台灣資深室內設計顧問,擅長把屋主的「夢想」翻譯成「預算現實」。
你了解 2026 年台灣裝修行情:
- 一般行情:每坪約 6-8 萬 (基礎風)、8-12 萬 (中等)、12-18 萬 (品味質感)、18 萬+ (奢華)
- 中島廚房:單做約 8-15 萬
- 更衣室:5-12 萬
- 系統櫃 vs 木作:系統便宜 30-50%
- 木地板 vs SPC:木地板貴 50-100%
- 風格成本由低到高:鄉村 → 現代簡約 ≈ 北歐 → 工業 → 日式 ≈ 混搭 → 輕奢 → 古典歐式

你的口吻:溫暖務實、直接但不嚇人、給具體數字。
"""


def _user_prompt(area: float, rooms: int, budget: int, style: str,
                 has_kids: str, must_haves: list[str]) -> str:
    must_haves_str = "、".join(must_haves) if must_haves else "(無特別要求)"
    return f"""屋主的條件:
- 坪數: {area} 坪
- 房間數: {rooms} 房
- 預算: {budget} 萬 (含設計費、工程、家具家電)
- 偏好風格: {style}
- 家庭成員: 夫妻 + {has_kids}
- 必備功能: {must_haves_str}

請以 JSON 回傳 (純 JSON,不要 markdown fence):

{{
  "achievement_pct": 75,
  "verdict_one_line": "你的預算可以做到 75% 的{style},需要在某些細節上妥協,但整體效果不會差。",
  "budget_breakdown": [
    {{"category": "基礎工程 (拆除/水電/泥作)", "amount": 50}},
    {{"category": "木作/系統櫃", "amount": 60}},
    {{"category": "風格表現 (磁磚/油漆/燈具)", "amount": 30}},
    {{"category": "家具家電", "amount": 35}},
    {{"category": "設計費", "amount": 25}}
  ],
  "must_compromise": [
    {{"item": "中島廚房改一字型", "save_amount": 12, "reason": "中島施工費 + 五金較貴,一字型可省 60%"}},
    {{"item": "全室木地板改 SPC", "save_amount": 8, "reason": "SPC 質感接近木地板但便宜近半"}},
    {{"item": "..."}}
  ],
  "can_keep": [
    "風格主軸 (75% 還原度)",
    "..."
  ],
  "designer_advice": "找擅長系統櫃 + 軟裝搭配的設計師,他們能用較少木作達到風格效果。建議找做過{style}小坪數案例的人。"
}}

注意:
- achievement_pct 是 0-100 整數
- budget_breakdown 加總應該約等於屋主預算 ({budget} 萬)
- must_compromise 至少 3 項,每項 save_amount 是省下的金額 (萬)
- can_keep 至少 3 項,具體可保留的部分
- 所有金額以「萬」為單位
"""


def _gauge_chart(pct: int, style: str) -> Image.Image:
    """畫達成度儀表板。"""
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=110, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    # 半圓 gauge
    theta1 = np.pi          # 180°
    theta2 = np.pi - (np.pi * pct / 100)  # 依達成度
    # 灰底
    ax.barh(0.5, theta1, left=0, height=0.6, color="#e2e8f0", alpha=0.8)
    # 漸層填色 (依達成度配色)
    color = "#22c55e" if pct >= 80 else ("#3b82f6" if pct >= 60 else ("#f59e0b" if pct >= 40 else "#ef4444"))
    ax.barh(0.5, theta1 - theta2, left=theta2, height=0.6, color=color)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, np.pi)
    ax.set_theta_offset(0)
    ax.set_theta_direction(1)
    ax.axis("off")
    # 中央百分比 + 風格
    ax.text(np.pi / 2, 0, f"{pct}%", ha="center", va="center",
            fontsize=44, fontweight="bold", color=color)
    ax.text(np.pi / 2, -0.25, f"{style} 達成度", ha="center", va="center",
            fontsize=12, color="#475569")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _budget_chart(breakdown: list[dict], total: int) -> Image.Image:
    """畫預算分配 horizontal bar chart。"""
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    items = breakdown
    categories = [it["category"] for it in items]
    amounts = [float(it["amount"]) for it in items]
    colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4", "#84cc16"][:len(items)]
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, amounts, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("金額 (萬)", fontsize=10, color="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    # 在 bar 末標數字
    for bar, amt in zip(bars, amounts):
        ax.text(bar.get_width() + max(amounts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{amt:.0f} 萬", va="center", fontsize=10, color="#0f172a", fontweight="600")
    ax.set_xlim(0, max(amounts) * 1.18)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def analyze(area: float, rooms: int, budget: int, style: str, has_kids: str, must_haves: list[str]):
    print(f"[budget_therapist] analyze  area={area} rooms={rooms} budget={budget} style={style}", flush=True)
    if not area or not budget or not style:
        return None, None, "請填齊坪數、預算與風格", "", "", ""
    try:
        client = _claude()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(area, rooms, budget, style, has_kids, must_haves)}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rstrip("`").strip()
        data = json.loads(text)
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, f"❌ 分析失敗:{e}", "", "", ""

    pct = int(data.get("achievement_pct", 0))
    gauge = _gauge_chart(pct, style)
    breakdown = data.get("budget_breakdown", [])
    bar = _budget_chart(breakdown, budget) if breakdown else None

    # 一句話判決
    verdict = f"### 🎯 {data.get('verdict_one_line', '')}\n"

    # 妥協清單
    must_comp = data.get("must_compromise", [])
    comp_md = "### ⚖️ 建議妥協 (但能省錢)\n\n"
    total_save = 0
    for c in must_comp:
        save = c.get("save_amount", 0)
        total_save += save
        comp_md += f"- **{c.get('item','')}** — 省 **{save} 萬**\n  > {c.get('reason','')}\n\n"
    if must_comp:
        comp_md += f"\n**💰 總共可省約 {total_save} 萬**"

    # 保留清單
    keep = data.get("can_keep", [])
    keep_md = "### ✅ 你可以保留的部分\n\n" + "\n".join(f"- {k}" for k in keep)

    # 設計師建議
    advice = data.get("designer_advice", "")
    advice_md = f"### 👤 找設計師時的建議\n\n{advice}"

    return gauge, bar, verdict, comp_md, keep_md, advice_md


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="💸",
            title="預算焦慮治療師",
            subtitle="把模糊的「我有 X 萬能做到嗎?」變成清楚的達成度 + 妥協清單 + 省錢方案,讓裝修決策不再焦慮",
            tools=[
                ("Claude Sonnet 4.6", "看 5 個條件給整合分析,生成達成度 + 妥協建議"),
                ("台灣裝修行情知識庫", "Claude 內建 2026 台灣每坪/每項目市場價"),
                ("matplotlib", "達成度儀表 + 預算分配 bar chart 視覺化"),
            ],
            cost="~$0.05",
            cost_detail="單次 Claude API call",
            time="5-10 秒",
            time_detail="一次 LLM call + 圖表渲染",
            badges=["決策助手", "視覺化", "台灣行情"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:填好下方 5 個條件,AI 用台灣 2026 市場價分析你的預算 — 能做到幾成?哪裡可以省?哪些一定要花?最後一份報告書級別的決策建議。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                area = gr.Number(label="坪數 (室內實坪)", value=25, minimum=5, maximum=200)
                rooms = gr.Number(label="房間數", value=3, minimum=1, maximum=10, precision=0)
                budget = gr.Number(label="預算 (萬,含設計+工程+家具家電)", value=200, minimum=30, maximum=2000)
                style = gr.Dropdown(label="想要的風格", choices=STYLE_OPTIONS, value="北歐風")
                has_kids = gr.Radio(label="家庭成員", choices=HAS_KIDS_OPTIONS, value="無")
                must_haves = gr.CheckboxGroup(label="必備功能 (選填,可複選)", choices=MUST_HAVE_OPTIONS)
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("💸 開始診斷", variant="primary", scale=2)
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                with gr.Row():
                    gauge_view = gr.Image(label="達成度", type="pil", height=240, interactive=False)
                    bar_view = gr.Image(label="預算分配", type="pil", height=240, interactive=False)
                verdict_md = gr.Markdown()
                comp_md = gr.Markdown()
                keep_md = gr.Markdown()
                advice_md = gr.Markdown()

        btn.click(
            analyze,
            inputs=[area, rooms, budget, style, has_kids, must_haves],
            outputs=[gauge_view, bar_view, verdict_md, comp_md, keep_md, advice_md],
        )

    return demo
