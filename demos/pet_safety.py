"""🐕 寵物友善設計檢測 — 上傳家裡照片,AI 為毛孩體檢居家安全。

8 大維度:
  · 尖角危險 (家具邊角、玻璃)
  · 有毒植物 (鴨腳木、百合、龜背芋等對貓狗有毒)
  · 易碎 / 易倒物 (花瓶、立燈、相框)
  · 線材外露 (電線、充電線)
  · 收納安全 (清潔劑、藥品)
  · 逃脫風險 (窗戶、陽台)
  · 摩擦地板 (光滑磁磚對老狗)
  · 寵物動線

回傳:總分 + 各維度評分 + 具體改善建議
"""

from __future__ import annotations

import base64
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


CHECK_DIMENSIONS = [
    ("尖角危險",   "sharp_corners",   "家具邊角、玻璃桌、金屬框,寵物追逐時可能撞傷"),
    ("有毒植物",   "toxic_plants",    "百合、鴨腳木、龜背芋、聖誕紅等對貓狗有毒"),
    ("易碎物品",   "fragile_items",   "花瓶、立燈、相框等寵物可能撞倒打破的物件"),
    ("線材外露",   "exposed_cables",  "電線、充電線、網路線,寵物啃咬風險"),
    ("收納安全",   "storage_hazards", "清潔劑、藥品、人類食物 (巧克力等) 是否收好"),
    ("逃脫風險",   "escape_risk",     "窗戶開口、陽台縫隙、紗門間距"),
    ("地板摩擦",   "floor_traction",  "光滑磁磚對老狗膝蓋傷害大,地毯加分"),
    ("寵物動線",   "pet_flow",        "寵物有沒有專屬休息點、能否安全到食盆/廁所"),
]


def _claude_client():
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY 未設定")
    return Anthropic(api_key=key)


def _img_b64(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


PROMPT = """你是寵物友善設計顧問,專門評估居家環境對寵物的安全性。

請看著這張室內照片,為以下 8 個維度評分 (0-10 分,10 為最安全),並給具體觀察 + 改善建議:

維度 (key):
- sharp_corners: 尖角危險
- toxic_plants: 有毒植物 (對貓狗有毒的植物)
- fragile_items: 易碎易倒物
- exposed_cables: 線材外露
- storage_hazards: 收納安全 (清潔劑/藥品)
- escape_risk: 逃脫風險 (窗/陽台/紗門)
- floor_traction: 地板摩擦力
- pet_flow: 寵物動線設計

回傳純 JSON (不要 markdown fence):
{
  "overall_score": 75,
  "verdict_one_line": "整體尚可,但有 3 個地方需要立即處理",
  "dimensions": {
    "sharp_corners":   {"score": 6, "observation": "...", "advice": "..."},
    "toxic_plants":    {"score": 8, "observation": "看不到明顯植物", "advice": "..."},
    "fragile_items":   {"score": 5, "observation": "...", "advice": "..."},
    "exposed_cables":  {"score": 4, "observation": "...", "advice": "..."},
    "storage_hazards": {"score": 9, "observation": "...", "advice": "..."},
    "escape_risk":     {"score": 7, "observation": "...", "advice": "..."},
    "floor_traction":  {"score": 5, "observation": "...", "advice": "..."},
    "pet_flow":        {"score": 6, "observation": "...", "advice": "..."}
  },
  "top_priorities": [
    {"action": "處理外露線材", "urgency": "高", "cost": "< 500 元"},
    ...3-5 個最重要的改善
  ]
}

評分原則:
- 9-10:完全安全
- 7-8:基本 OK,少數小調整
- 5-6:有風險,需注意
- 3-4:明顯危險,該改善
- 0-2:緊急
"""


def _gauge(score: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=110, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    theta_full = np.pi
    theta_filled = np.pi * (score / 100)
    ax.barh(0.5, theta_full, left=0, height=0.6, color="#e2e8f0")
    color = "#22c55e" if score >= 80 else "#3b82f6" if score >= 60 else "#f59e0b" if score >= 40 else "#ef4444"
    ax.barh(0.5, theta_filled, left=np.pi - theta_filled, height=0.6, color=color)
    ax.set_ylim(0, 1.2); ax.set_xlim(0, np.pi)
    ax.set_theta_offset(0); ax.set_theta_direction(1)
    ax.axis("off")
    ax.text(np.pi / 2, 0, f"{score}", ha="center", va="center", fontsize=42, fontweight="bold", color=color)
    ax.text(np.pi / 2, -0.22, "/ 100 寵物友善分數", ha="center", va="center", fontsize=11, color="#475569")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _radar(dim_scores: dict) -> Image.Image:
    labels = [d[0] for d in CHECK_DIMENSIONS]
    keys = [d[1] for d in CHECK_DIMENSIONS]
    values = [dim_scores.get(k, {}).get("score", 0) for k in keys]
    values_closed = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=110, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#f8fafc")
    ax.plot(angles_closed, values_closed, color="#3b82f6", linewidth=2.5)
    ax.fill(angles_closed, values_closed, alpha=0.22, color="#3b82f6")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10, color="#1e293b")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#94a3b8")
    ax.tick_params(axis="x", pad=12)
    ax.spines["polar"].set_color("#cbd5e1")
    ax.grid(color="#cbd5e1", alpha=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def analyze(image: Image.Image, pet_type: str):
    print(f"[pet_safety] analyze  pet_type={pet_type}", flush=True)
    if image is None:
        return None, None, "請先上傳家裡照片", "", ""
    try:
        client = _claude_client()
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _img_b64(image)}},
                    {"type": "text", "text": PROMPT + f"\n\n屋主養的是: {pet_type}\n請特別考慮這個寵物類型的常見風險。"},
                ],
            }],
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
        return None, None, f"❌ 失敗:{e}", "", ""

    overall = int(data.get("overall_score", 0))
    dims = data.get("dimensions", {})
    gauge_img = _gauge(overall)
    radar_img = _radar(dims)

    verdict = f"### 🎯 {data.get('verdict_one_line', '')}\n"

    # 各維度詳細表
    lines = ["### 📋 各維度詳細評估\n"]
    lines.append("| 維度 | 分數 | 觀察 | 建議 |\n|---|---|---|---|\n")
    for name, key, _ in CHECK_DIMENSIONS:
        d = dims.get(key, {})
        score = d.get("score", 0)
        emoji = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
        obs = (d.get("observation", "") or "")[:80]
        adv = (d.get("advice", "") or "")[:100]
        lines.append(f"| {name} | {emoji} **{score}**/10 | {obs} | {adv} |\n")
    details_md = "".join(lines)

    # Top priorities
    priorities = data.get("top_priorities", [])
    if priorities:
        prio_md = "### 🚨 優先處理 (依緊急度)\n\n"
        for i, p in enumerate(priorities, 1):
            urgency = p.get("urgency", "中")
            emoji = "🔴" if urgency == "高" else "🟡" if urgency == "中" else "🟢"
            cost = p.get("cost", "")
            prio_md += f"{i}. {emoji} **{p.get('action','')}** — 緊急度 {urgency} · 成本 {cost}\n"
    else:
        prio_md = ""

    return gauge_img, radar_img, verdict, details_md, prio_md


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🐕",
            title="寵物友善設計檢測",
            subtitle="上傳家裡照片,AI 為毛孩體檢居家安全 — 8 個維度評分 + 具體改善建議,讓家對你也對毛孩友善",
            tools=[
                ("Claude Sonnet 4.6 (vision)", "看圖 + 寵物類型,8 維度結構化評估"),
                ("matplotlib radar + gauge", "雷達圖 + 儀表板視覺化"),
                ("台灣常見寵物風險知識庫", "Claude 內建有毒植物 / 物品清單"),
            ],
            cost="$0.05",
            cost_detail="一次 Claude vision call",
            time="10-15 秒",
            time_detail="Claude 評估 + 圖表渲染",
            badges=["利基市場", "視覺評估", "養寵家庭"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:養狗養貓的家庭必玩!上傳家裡照片,AI 從 8 個維度 (尖角/有毒植物/線材/逃脫風險...) 評估對寵物的安全性,給具體改造建議。<br/>📌 適合:剛養寵物的家庭、租屋族、家有老狗或幼貓。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="上傳家裡照片", height=320)
                pet_type = gr.Radio(
                    label="家裡的寵物是?",
                    choices=["小型犬 (柴犬等)", "中大型犬", "貓咪", "兔子 / 鼠類", "鳥類", "魚 / 爬蟲", "多種寵物"],
                    value="貓咪",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🐾 檢測寵物友善度 (~15s)", variant="primary", scale=2)
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                with gr.Row():
                    gauge_view = gr.Image(label="整體寵物友善度", type="pil", height=240, interactive=False)
                    radar_view = gr.Image(label="8 維度雷達圖", type="pil", height=320, interactive=False)
                verdict_md = gr.Markdown()
                details_md = gr.Markdown()
                prio_md = gr.Markdown()

        btn.click(
            analyze, inputs=[in_img, pet_type],
            outputs=[gauge_view, radar_view, verdict_md, details_md, prio_md],
        )

    return demo
