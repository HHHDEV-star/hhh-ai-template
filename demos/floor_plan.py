"""📐 AI 平面圖生成器 — 把條件變成可看的 2D 平面圖。

流程:
  1. 屋主填:坪數 / 房數 / 家庭成員 / 必備功能 / 風格
  2. Claude (室內設計師角色) 規劃 layout → 結構化 JSON (房間位置/家具)
  3. matplotlib 把 JSON → 2D 平面圖 (牆 + 房間色塊 + 家具圖示 + 標籤)
  4. Claude 同時生:設計考量、潛在問題、2 個替代方案
"""

from __future__ import annotations

import io
import json
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
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


# 房間類型 → 顏色 + 中文標籤
ROOM_TYPES = {
    "entrance":  {"color": "#fce7f3", "label": "玄關"},
    "living":    {"color": "#dbeafe", "label": "客廳"},
    "dining":    {"color": "#fef3c7", "label": "餐廳"},
    "kitchen":   {"color": "#fed7aa", "label": "廚房"},
    "master":    {"color": "#e0e7ff", "label": "主臥"},
    "bedroom":   {"color": "#ddd6fe", "label": "臥室"},
    "kids":      {"color": "#fef9c3", "label": "兒童房"},
    "study":     {"color": "#d1fae5", "label": "書房"},
    "bathroom":  {"color": "#cffafe", "label": "衛浴"},
    "balcony":   {"color": "#bbf7d0", "label": "陽台"},
    "storage":   {"color": "#fed7d7", "label": "儲藏室"},
    "closet":    {"color": "#f3e8ff", "label": "更衣室"},
    "laundry":   {"color": "#e5e7eb", "label": "洗衣間"},
    "corridor":  {"color": "#f1f5f9", "label": "走道"},
    "other":     {"color": "#e5e7eb", "label": "其他"},
}

# 家具圖示 (用 emoji 配抽象矩形)
FURNITURE_ICONS = {
    "sofa":     "🛋️", "tv": "📺",  "coffee_table": "▣", "dining_table": "◫",
    "bed":      "🛏️", "wardrobe": "▦", "desk": "▭",  "chair": "■",
    "kitchen_island": "▬", "stove": "🔥", "sink": "🚰", "fridge": "📦",
    "toilet":   "🚽", "shower": "🚿",  "bath": "🛁", "washbasin": "◯",
    "shoe_cabinet": "▤", "shelf": "▥",
}


def _claude_client():
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY 未設定")
    return Anthropic(api_key=key)


SYSTEM_PROMPT = """你是台灣資深室內設計師,擅長把屋主需求轉成 2D 平面圖。

座標系統:
- 整個房子是 10x10 的 grid (x 從 0-10 向右,y 從 0-10 向上)
- 每個房間是一個矩形:{"x": 起點, "y": 起點, "w": 寬, "h": 高}
- 不同房間不可重疊
- 房間總和應該等於屋主給的坪數 (約略,1 grid 單位約 0.3 坪參考)
- 動線:玄關通常在 (0, 0) 附近,公領域連通,私領域往內

可用 room type:
  entrance / living / dining / kitchen / master / bedroom / kids /
  study / bathroom / balcony / storage / closet / laundry / corridor

家具類型:
  sofa, tv, coffee_table, dining_table, bed, wardrobe, desk, chair,
  kitchen_island, stove, sink, fridge, toilet, shower, bath, washbasin,
  shoe_cabinet, shelf
"""


def _user_prompt(area: float, rooms_str: str, family: str, must_haves: list[str], style: str) -> str:
    must_str = "、".join(must_haves) if must_haves else "(無特別要求)"
    return f"""屋主條件:
- 坪數: {area} 坪
- 房數: {rooms_str}
- 家庭成員: {family}
- 必備功能: {must_str}
- 偏好風格: {style}

請設計一個合理的 2D 平面圖規劃,以**純 JSON**回傳 (不要 markdown fence):

{{
  "rooms": [
    {{
      "name": "客廳",
      "type": "living",
      "x": 0.5, "y": 0.5, "w": 3.5, "h": 2.5,
      "furniture": [
        {{"type": "sofa", "x": 0.8, "y": 0.7, "w": 1.5, "h": 0.5}},
        {{"type": "tv", "x": 2.5, "y": 3.0, "w": 1.2, "h": 0.2}},
        {{"type": "coffee_table", "x": 1.5, "y": 1.5, "w": 0.8, "h": 0.5}}
      ]
    }},
    ...
  ],
  "design_notes": [
    "公領域開放式設計促進家人互動",
    "主臥與小孩房相鄰方便照顧",
    "..."
  ],
  "potential_issues": [
    "若預算緊,廚房中島可改一字型",
    "..."
  ],
  "alternatives": [
    "若不想要書房,改成第二客房 (+5 坪彈性)",
    "..."
  ]
}}

重要:
- 房間座標必須在 0-10 範圍內
- 房間不可重疊
- 家具座標在所屬房間內 (相對於整體 grid)
- 必須包含至少:玄關、客廳、餐廳、廚房、衛浴、主臥
- 房間數要符合屋主要求
- 必備功能必須出現
"""


# ===== matplotlib renderer =====
def _render_plan(plan: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(11, 9), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rooms = plan.get("rooms", [])
    if not rooms:
        ax.text(5, 5, "(無房間資料)", ha="center", va="center", fontsize=20, color="#94a3b8")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    else:
        # 1. 畫房間色塊 + 邊框
        for r in rooms:
            color = ROOM_TYPES.get(r.get("type", "other"), ROOM_TYPES["other"])["color"]
            label = ROOM_TYPES.get(r.get("type", "other"), ROOM_TYPES["other"])["label"]
            name = r.get("name", label)
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            rect = FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  linewidth=2.5, edgecolor="#1e293b",
                                  facecolor=color, alpha=0.85)
            ax.add_patch(rect)
            # 房間名稱
            ax.text(x + w / 2, y + h - 0.18, name, ha="center", va="top",
                    fontsize=13, fontweight="700", color="#0f172a")
            # 房間尺寸標籤
            ax.text(x + w / 2, y + 0.1, f"{w:.1f} × {h:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#64748b", style="italic")

            # 2. 畫家具
            for f in r.get("furniture", []):
                fx, fy = f.get("x", x), f.get("y", y)
                fw, fh = f.get("w", 0.3), f.get("h", 0.3)
                ftype = f.get("type", "")
                icon = FURNITURE_ICONS.get(ftype, "■")
                # 家具背景小框
                fbox = Rectangle((fx, fy), fw, fh,
                                 linewidth=1, edgecolor="#475569",
                                 facecolor="white", alpha=0.7)
                ax.add_patch(fbox)
                # 圖示文字
                ax.text(fx + fw / 2, fy + fh / 2, icon,
                        ha="center", va="center", fontsize=12)

        # 3. 外框 + 北向
        ax.text(5, 10.3, "↑ N", ha="center", fontsize=14, fontweight="600", color="#0f172a")
        ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.8)
        ax.set_aspect("equal")
        ax.axis("off")

    # legend
    legend_handles = []
    used_types = set(r.get("type") for r in rooms)
    for t in ["living", "dining", "kitchen", "master", "bedroom", "study", "kids", "bathroom", "balcony", "closet"]:
        if t in used_types:
            cfg = ROOM_TYPES[t]
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=cfg["color"], edgecolor="#1e293b", label=cfg["label"]))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 5),
                  bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ===== Handler =====
def generate(area: float, rooms_str: str, family: str, must_haves: list[str], style: str):
    print(f"[floor_plan] generate  area={area} rooms={rooms_str} family={family}", flush=True)
    if not area or not rooms_str:
        return None, "請填齊坪數與房數", "", ""

    client = _claude_client()
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(area, rooms_str, family, must_haves, style)}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rstrip("`").strip()
        plan = json.loads(text)
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ Claude 規劃失敗:{e}", "", ""

    try:
        img = _render_plan(plan)
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ 平面圖渲染失敗:{e}", "", ""

    notes_md = "### 🎯 設計考量\n\n" + "\n".join(f"- {n}" for n in plan.get("design_notes", []))
    issues_md = "### ⚠️ 注意事項\n\n" + "\n".join(f"- {i}" for i in plan.get("potential_issues", []))
    alts_md = "### 💡 替代方案\n\n" + "\n".join(f"- {a}" for a in plan.get("alternatives", []))

    summary = f"✓ 完成 · **{area} 坪 · {rooms_str}** · {len(plan.get('rooms', []))} 個房間"
    detail = f"{notes_md}\n\n{issues_md}\n\n{alts_md}"

    return img, summary, detail, json.dumps(plan, ensure_ascii=False, indent=2)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="📐",
            title="AI 平面圖生成器",
            subtitle="用講的就能畫平面圖 — 屋主填條件,Claude 規劃 layout,即時看見 2D 平面圖 + 設計考量",
            tools=[
                ("Claude Sonnet 4.6", "規劃 layout JSON,撰寫設計考量"),
                ("matplotlib + FancyBboxPatch", "渲染 2D 平面圖 (房間/家具/標籤)"),
                ("中文字型 + emoji 圖示", "家具用 emoji 抽象表示"),
            ],
            cost="$0.02-0.03",
            cost_detail="只一次 Claude 規劃",
            time="10-15 秒",
            time_detail="Claude 規劃 + matplotlib 渲染",
            badges=["決策工具", "輕量", "可互動"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:填好坪數、房數、家庭成員、必備功能,Claude 用台灣裝修常識規劃 layout,matplotlib 即時畫出 2D 平面圖。<br/>📌 適合:屋主買房前模擬、設計師快速產初稿、跟家人討論格局。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                area = gr.Number(label="坪數 (室內實坪)", value=30, minimum=8, maximum=200)
                rooms_str = gr.Textbox(
                    label="房數需求",
                    placeholder="例:3房2廳1廚2衛",
                    value="3房2廳1廚2衛",
                )
                family = gr.Radio(
                    label="家庭成員",
                    choices=["單身", "夫妻", "夫妻+1孩", "夫妻+2孩", "三代同堂"],
                    value="夫妻+1孩",
                )
                must_haves = gr.CheckboxGroup(
                    label="必備功能 (可多選)",
                    choices=["中島廚房", "更衣室", "書房", "玄關櫃", "陽台", "儲藏室", "洗衣間", "雙衛浴"],
                    value=["玄關櫃", "陽台"],
                )
                style = gr.Dropdown(
                    label="偏好風格",
                    choices=["北歐極簡", "現代簡約", "輕奢風", "日式禪風", "工業風", "混搭風"],
                    value="北歐極簡",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("📐 生成平面圖 (~15s)", variant="primary", scale=2)
                summary_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 2D 平面圖", elem_classes=["demo-section-title"])
                plan_img = gr.Image(label=None, show_label=False, height=480, type="pil", interactive=False)
                detail_md = gr.Markdown()
                with gr.Accordion("🔬 原始 JSON layout (給工程師看)", open=False):
                    raw_json = gr.Code(language="json", label=None)

        btn.click(
            generate,
            inputs=[area, rooms_str, family, must_haves, style],
            outputs=[plan_img, summary_md, detail_md, raw_json],
        )

    return demo
