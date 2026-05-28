"""☀️ 自然光全天追蹤 — 物理計算太陽位置,模擬全天日照軌跡。

輸入:房子坐向 + 樓層 + 周邊建物高度 (選填)
       ↓
物理計算 (用太陽軌跡公式,以台灣 24°N 為基準):
   清晨 6:00 / 上午 9:00 / 中午 12:00 / 下午 15:00 / 傍晚 18:00
       ↓
matplotlib 畫光線示意圖 (5 個時段 + 全天動畫)
       ↓
Claude 給「家具該擺哪」的建議
"""

from __future__ import annotations

import io
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.patches import Rectangle, Wedge
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


# 台灣 (北回歸線附近) 緯度
TAIWAN_LAT = 24.0

# 簡化太陽位置 (春秋分,夏冬會偏移約 ±23.4°)
TIME_SOLAR = {
    "06:00": {"altitude": 8,  "azimuth": -90,  "label": "清晨"},   # 東偏北
    "09:00": {"altitude": 40, "azimuth": -55,  "label": "上午"},
    "12:00": {"altitude": 75, "azimuth": 0,    "label": "正午"},   # 正南
    "15:00": {"altitude": 50, "azimuth": 55,   "label": "下午"},
    "18:00": {"altitude": 12, "azimuth": 90,   "label": "傍晚"},   # 西偏北
}

ORIENTATIONS = {
    "南向 (光線最佳)":    0,
    "東向 (晨光好)":       -90,
    "東南向":             -45,
    "西向 (西曬強)":       90,
    "西南向":              45,
    "北向 (光線弱)":      180,
}


def _make_room_plot(orientation_deg: float, time_key: str, floor_height: int, neighbor_height: int) -> Image.Image:
    """畫一張房間 + 光線示意圖。orientation_deg: 房子正面朝向 (0=南)。"""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    sun_data = TIME_SOLAR[time_key]
    # 太陽方位角扣掉房子朝向 = 光對房子的相對角度
    sun_relative_az = sun_data["azimuth"] - orientation_deg
    altitude = sun_data["altitude"]

    # 鄰居建物對採光的遮蔽:鄰居越高、自家樓層越低,遮蔽角度越大
    # 簡化:對方擋掉 altitude 角度低於 (neighbor_h - my_floor*3) / distance 的光
    shadow_angle = max(0, (neighbor_height - floor_height * 3) * 8.0 / 10.0)
    light_strength = max(0, min(1, (altitude - shadow_angle) / 70))

    # 1. 畫房間 (中間 8x6 矩形)
    room = Rectangle((1, 1), 8, 6, linewidth=2.5, edgecolor="#0f172a", facecolor="white")
    ax.add_patch(room)
    # 牆內陰影 = 沒光的地方
    ax.text(5, 4, f"{floor_height} 樓\n房間", ha="center", va="center", fontsize=14, fontweight="600", color="#475569")

    # 2. 光線從哪邊射入
    if sun_relative_az != 0 and -180 < sun_relative_az < 180:
        # 確定光來自哪一邊牆
        # sun_relative_az: -180 ~ 180,0 = 從正面南方,正 = 西方,負 = 東方
        rad = np.radians(sun_relative_az - 90)  # 轉成 matplotlib 角度 (0 = 右)
        sun_x = 5 + 8 * np.cos(rad)
        sun_y = 4 + 6 * np.sin(rad)
        # 確保太陽在外圍
        if -90 < sun_relative_az < 90:  # 南面採光 (主立面)
            entry_side = "south"
            for x_offset in np.linspace(1, 9, 8):
                ax.annotate('', xy=(x_offset, 1.05), xytext=(x_offset, 0.2),
                            arrowprops=dict(arrowstyle='->', color=f'#fbbf24', lw=1.5, alpha=0.4 + 0.5 * light_strength))
        elif sun_relative_az >= 90:  # 西側
            entry_side = "west"
            for y_offset in np.linspace(2, 6, 5):
                ax.annotate('', xy=(8.95, y_offset), xytext=(9.8, y_offset),
                            arrowprops=dict(arrowstyle='->', color=f'#fbbf24', lw=1.5, alpha=0.4 + 0.5 * light_strength))
        else:  # 東側
            entry_side = "east"
            for y_offset in np.linspace(2, 6, 5):
                ax.annotate('', xy=(1.05, y_offset), xytext=(0.2, y_offset),
                            arrowprops=dict(arrowstyle='->', color=f'#fbbf24', lw=1.5, alpha=0.4 + 0.5 * light_strength))
    else:
        entry_side = "north"

    # 3. 畫光照亮的區域 (淡黃色 fill)
    sun_color = "#fef3c7" if light_strength > 0.6 else "#e2e8f0"
    intensity_fill = light_strength
    if entry_side == "south":
        ax.add_patch(Rectangle((1, 1), 8, 2.5, facecolor="#fbbf24", alpha=0.15 * intensity_fill))
    elif entry_side == "west":
        ax.add_patch(Rectangle((6.5, 1), 2.5, 6, facecolor="#fbbf24", alpha=0.15 * intensity_fill))
    elif entry_side == "east":
        ax.add_patch(Rectangle((1, 1), 2.5, 6, facecolor="#fbbf24", alpha=0.15 * intensity_fill))

    # 4. 鄰居建物 (如果有)
    if neighbor_height > 0:
        # 畫在房子下方 (代表南面遮蔽) 如果朝南
        neighbor_y_top = max(0, min(1, (neighbor_height - floor_height * 3) * 0.02))
        n = Rectangle((1, -1.2 + neighbor_y_top), 8, 0.3,
                      facecolor="#94a3b8", edgecolor="#475569", alpha=0.7)
        ax.add_patch(n)
        ax.text(5, -1, f"鄰居 {neighbor_height}m", ha="center", fontsize=8, color="#475569")

    # 5. 太陽位置示意 (頂部)
    sun_x_top = 5 + (sun_relative_az / 90) * 4
    sun_y_top = 7.2 + altitude / 30
    ax.scatter([sun_x_top], [sun_y_top], s=600, c="#fbbf24", edgecolors="#f59e0b", linewidth=2.5, zorder=5)
    ax.text(sun_x_top, sun_y_top, "☀", ha="center", va="center", fontsize=22, zorder=6)

    # 6. 標題 + 光強百分比
    intensity_pct = int(light_strength * 100)
    ax.text(5, 8.3, f"{time_key} · {sun_data['label']}", ha="center", fontsize=15, fontweight="700", color="#0f172a")
    ax.text(5, 7.8, f"自然光強度 {intensity_pct}%", ha="center", fontsize=11, color="#475569")

    # 7. 北向箭頭
    ax.annotate('N', xy=(0.5, 7), fontsize=10, color="#0f172a", fontweight="700")
    ax.annotate('', xy=(0.5, 7), xytext=(0.5, 6.4),
                arrowprops=dict(arrowstyle='->', color="#0f172a", lw=2))

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.8, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _claude_advice(orientation_label: str, floor_height: int, neighbor_height: int, results: dict) -> str:
    """根據 5 個時段的光照數據,Claude 給家具擺放建議。"""
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "_(Claude 未設定,跳過建議)_"

    intensities = "\n".join(f"- {t} ({TIME_SOLAR[t]['label']}): 光強 {results[t]}%" for t in TIME_SOLAR)
    prompt = f"""你是台灣居家規劃顧問,根據以下自然光分析給家具擺放建議。

房屋條件:
- 朝向: {orientation_label}
- 樓層: {floor_height} 樓 (約 {floor_height * 3} 米高)
- 鄰近建物高度: {neighbor_height} 米

全天光強分布:
{intensities}

請寫一份 250-350 字的「家具配置建議」(繁中、markdown 格式),包含:
1. **採光特性總結** (一段)
2. **最佳擺放位置** (沙發/床/工作桌/植物各該擺哪)
3. **要避免的位置** (哪邊容易曬到家具褪色 / 過熱 / 缺光)
4. **採光優化建議** (窗簾類型 / 反光面 / 鏡面位置)
口吻務實。
"""
    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def analyze(orientation: str, floor: int, neighbor: int):
    print(f"[sunlight] analyze  orientation={orientation} floor={floor} neighbor={neighbor}", flush=True)
    if not orientation or floor is None:
        return None, None, None, None, None, "請填齊朝向跟樓層", ""

    deg = ORIENTATIONS.get(orientation, 0)

    # 算每個時段的光強 (給 Claude 用)
    results_pct = {}
    for t, data in TIME_SOLAR.items():
        shadow_angle = max(0, (neighbor - floor * 3) * 8.0 / 10.0)
        light_strength = max(0, min(1, (data["altitude"] - shadow_angle) / 70))
        # 朝向反向會打折
        sun_rel_az = data["azimuth"] - deg
        if abs(sun_rel_az) > 120:  # 背向
            light_strength *= 0.3
        elif abs(sun_rel_az) > 90:
            light_strength *= 0.7
        results_pct[t] = int(light_strength * 100)

    # 5 張圖
    imgs = [_make_room_plot(deg, t, floor, neighbor) for t in TIME_SOLAR]
    img_06, img_09, img_12, img_15, img_18 = imgs

    summary = f"### ☀️ 採光摘要\n\n**朝向**:{orientation} · **{floor} 樓** · 鄰建物 **{neighbor} 米**\n\n"
    summary += "| 時段 | 光強 |\n|---|---|\n"
    for t, pct in results_pct.items():
        emoji = "🟢" if pct >= 70 else "🟡" if pct >= 40 else "🔴"
        summary += f"| {t} ({TIME_SOLAR[t]['label']}) | {emoji} **{pct}%** |\n"

    try:
        advice = _claude_advice(orientation, floor, neighbor, results_pct)
    except Exception as e:
        advice = f"_❌ 建議生成失敗: {e}_"

    return img_06, img_09, img_12, img_15, img_18, summary, advice


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="☀️",
            title="自然光全天追蹤",
            subtitle="物理計算太陽軌跡,模擬全天光線變化 — 買房規劃前先看「哪邊放沙發採光最好」",
            tools=[
                ("Solar position 物理計算", "依台灣緯度 24°N 算 5 時段太陽位置"),
                ("matplotlib geometry", "視覺化光線進入角度、強度、鄰建物遮蔽"),
                ("Claude Sonnet 4.6", "解讀光照數據給家具擺放建議"),
            ],
            cost="$0.02",
            cost_detail="只 Claude 寫一次建議",
            time="~10 秒",
            time_detail="純本機物理計算 + 1 次 Claude",
            badges=["物理模擬", "決策工具", "0 圖也能跑"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:輸入房子坐向、樓層、鄰近建物高度,系統用太陽軌跡公式算全天 5 時段光線變化,Claude 給家具該擺哪、哪邊要避免的建議。<br/>📌 適合:看房前評估、買房後規劃、改造方案討論。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                orientation = gr.Radio(
                    label="房子正面朝向",
                    choices=list(ORIENTATIONS.keys()),
                    value="南向 (光線最佳)",
                )
                floor = gr.Number(label="樓層 (1-30+)", value=8, minimum=1, maximum=50, precision=0)
                neighbor = gr.Number(label="鄰近建物高度 (米,0 為空曠)", value=15, minimum=0, maximum=200)
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("☀️ 模擬全天光線 (~10s)", variant="primary", scale=2)
                summary_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 全天 5 時段光線變化", elem_classes=["demo-section-title"])
                with gr.Row():
                    img_06 = gr.Image(label="06:00", type="pil", height=220, interactive=False)
                    img_09 = gr.Image(label="09:00", type="pil", height=220, interactive=False)
                    img_12 = gr.Image(label="12:00", type="pil", height=220, interactive=False)
                with gr.Row():
                    img_15 = gr.Image(label="15:00", type="pil", height=220, interactive=False)
                    img_18 = gr.Image(label="18:00", type="pil", height=220, interactive=False)
                    gr.HTML('<div style="flex:1"></div>')  # filler
                gr.Markdown("### 🤖 家具擺放建議", elem_classes=["demo-section-title"])
                advice_md = gr.Markdown()

        btn.click(
            analyze, inputs=[orientation, floor, neighbor],
            outputs=[img_06, img_09, img_12, img_15, img_18, summary_md, advice_md],
        )

    return demo
