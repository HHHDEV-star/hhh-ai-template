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
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            ### 🎨 配色 DNA

            上傳一張圖,萃取主色 5-8 種(用 k-means clustering on RGB)。
            **用途範例**:幫每個案例自動算「主色指紋」,之後可以「按色相搜尋類似案例」。
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="上傳圖片", height=300)
                n = gr.Slider(3, 10, value=6, step=1, label="幾種主色")
                btn = gr.Button("分析", variant="primary")
            with gr.Column(scale=1):
                out_strip = gr.Image(type="pil", label="配色帶 (按佔比寬度)", height=120)
                out_table = gr.Markdown()

        btn.click(analyze, inputs=[inp, n], outputs=[out_strip, out_table])
        inp.change(analyze, inputs=[inp, n], outputs=[out_strip, out_table])
    return demo
