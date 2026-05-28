"""配色 DNA — 上傳一張圖,萃取主色調 (k-means clustering on RGB)。

設計用途:給室內設計案例萃出代表色,之後可以用色相搜尋類似案例。
這個 demo 不需要 ML 模型,純 NumPy + scikit-learn,適合當 boilerplate 範本。
"""

from __future__ import annotations

import io
from typing import List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def extract_palette(image: Image.Image, n_colors: int = 6) -> List[Tuple[Tuple[int, int, int], float]]:
    """回傳 [(rgb, weight), ...],weight 是該色佔比 (0..1),按佔比降冪排序。"""
    img = image.convert("RGB").copy()
    # 縮圖加速 (300px 邊長已足夠 k-means)
    img.thumbnail((300, 300))
    pixels = np.asarray(img).reshape(-1, 3)

    km = KMeans(n_clusters=n_colors, n_init="auto", random_state=42)
    km.fit(pixels)
    labels = km.labels_
    centers = km.cluster_centers_.astype(int)

    counts = np.bincount(labels, minlength=n_colors).astype(float)
    weights = counts / counts.sum()

    pairs = sorted(zip(centers, weights), key=lambda x: -x[1])
    return [(tuple(c), float(w)) for c, w in pairs]


def render_palette_strip(palette: List[Tuple[Tuple[int, int, int], float]], width: int = 800, height: int = 120) -> Image.Image:
    """畫一條按權重分配寬度的色帶。"""
    strip = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    x = 0
    for color, weight in palette:
        w = int(round(width * weight))
        draw.rectangle([x, 0, x + w, height], fill=color)
        # 在條上印 hex 文字 (黑或白看背景亮度)
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        if w > 60:
            draw.text((x + 8, height - 22), _rgb_to_hex(color), fill=text_color, font=font)
        x += w
    return strip


def analyze(image: Image.Image, n_colors: int):
    if image is None:
        return None, "請先上傳圖片"
    palette = extract_palette(image, n_colors=int(n_colors))
    strip = render_palette_strip(palette)

    lines = ["| 主色 | HEX | 佔比 |", "|---|---|---|"]
    for color, weight in palette:
        swatch = f"<span style='display:inline-block;width:18px;height:18px;background:{_rgb_to_hex(color)};border:1px solid #aaa;vertical-align:middle'></span>"
        lines.append(f"| {swatch} | `{_rgb_to_hex(color)}` | {weight*100:.1f}% |")
    return strip, "\n".join(lines)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🎨",
            title="配色 DNA",
            subtitle="把任何圖萃取出主色調 5-8 色,可作為案例的「色彩指紋」用於配色搜尋",
            tools=[
                ("scikit-learn KMeans", "RGB 空間 K-means 分群,找出代表色"),
                ("Pillow + NumPy", "影像縮圖、像素運算、配色條繪製"),
            ],
            cost="$0",
            cost_detail="純本機,無 API 依賴",
            time="< 1 秒",
            time_detail="300px 縮圖上跑 k-means",
            badges=["離線可用", "無 GPU"],
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳任何圖,AI 自動萃取出主色調 — 可作為案例的「色彩指紋」,或用來找配色相近的其他案例。</div>')
                inp = gr.Image(type="pil", label="上傳圖片", height=300)
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    n = gr.Slider(3, 10, value=6, step=1, label="萃取幾種主色",
                                  info="預設 6 色平衡,3-4 色更鮮明,8-10 色更細膩")
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🎨 萃取配色", variant="primary", scale=2)
            with gr.Column(scale=1, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 配色帶", elem_classes=["demo-section-title"])
                out_strip = gr.Image(type="pil", label="按佔比寬度排列", height=120)
                gr.Markdown("### 主色清單", elem_classes=["demo-section-title"])
                out_table = gr.Markdown()

        btn.click(analyze, inputs=[inp, n], outputs=[out_strip, out_table])
        inp.change(analyze, inputs=[inp, n], outputs=[out_strip, out_table])
    return demo
