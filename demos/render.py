"""🎨 AI 室內設計師 — 一張圖換 N 種風格。

兩種輸入模式 (使用者明確選擇):
  📐 線稿 / 平面圖 → SDXL + Canny ControlNet (lucataco/sdxl-controlnet)
                    輸入是黑白/灰階線條,保留每條線描邊,從 0 開始畫擬真
  📷 現場照片     → Interior Design SDXL + Depth ControlNet (rocketdigitalai/interior-design-sdxl)
                    輸入是彩色照片,保留 3D 結構但換家具/風格

統一 6 種風格 prompt,使用者體驗一致。
"""

from __future__ import annotations

import base64
import io
import os
import time

import gradio as gr
import requests
from PIL import Image


REPLICATE_API = "https://api.replicate.com/v1"
# 兩個後端模型
SKETCH_MODEL = "lucataco/sdxl-controlnet:06d6fae3b75ab68a28cd2900afa6033166910dd09fd9751047043a5bbb4c184b"
PHOTO_MODEL = "rocketdigitalai/interior-design-sdxl:a3c091059a25590ce2d5ea13651fab63f447f21760e50c358d4b850e844f59ee"

MODE_SKETCH = "📐 線稿 / 平面圖 / Sketchup 截圖"
MODE_PHOTO = "📷 現場照片 (空房或現有家具)"

# 風格 prompt — 兩種模式共用 (描述要的最終效果)
STYLES = {
    "🇸🇪 北歐極簡": "scandinavian minimalist interior, light oak wood, white walls, beige fabric sofa, soft natural daylight, plants, photorealistic, 8k",
    "🏭 工業風":   "industrial loft interior, exposed brick wall, concrete floor, edison bulb pendant lights, leather chesterfield sofa, metal pipes, dark wood accents, moody lighting",
    "🍵 日式禪風": "japanese zen interior, tatami platform, washi paper lamp, natural cedar wood, minimalist furniture, bonsai, warm soft light, serene",
    "💎 輕奢風":   "modern soft luxury interior, beige boucle sofa, brass accents, marble coffee table, designer floor lamp, neutral palette, elegant",
    "🌿 自然有機": "organic modern interior, rattan armchair, jute rug, large indoor plants, linen curtains, warm wood, biophilic design, natural light",
    "🎨 混搭風":   "eclectic interior, mid-century modern furniture, gallery wall, velvet sofa, brass and wood, vintage rug, layered lighting",
}

NEG_PROMPT_SKETCH = (
    "lowres, blurry, distorted, deformed, watermark, text, ugly, "
    "oversaturated, cartoon, painting, sketch, low quality"
)
NEG_PROMPT_PHOTO = "ugly, deformed, noisy, blurry, low quality, distorted, watermark, text, cartoon, illustration"


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError("REPLICATE_API_TOKEN 未設定")
    return t


def _to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    img = img.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.standard_b64encode(buf.getvalue()).decode()}"


def _replicate_run(version: str, inputs: dict, timeout: int = 300) -> list[str]:
    token = _token()
    h_post = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    h_get = {"Authorization": f"Bearer {token}"}
    body = {"version": version, "input": inputs}

    pred = None
    for attempt in range(6):
        r = requests.post(f"{REPLICATE_API}/predictions", headers=h_post, json=body, timeout=90)
        if r.status_code == 429:
            wait = int(r.json().get("retry_after", 6)) + 2
            print(f"[render] rate-limited, 等 {wait}s ({attempt+1}/6)", flush=True)
            time.sleep(wait)
            continue
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Replicate API {r.status_code}: {r.text[:300]}")
        pred = r.json()
        break
    if pred is None:
        raise RuntimeError("Replicate 持續 429")

    pred_id = pred["id"]
    status = pred["status"]
    deadline = time.time() + timeout
    while status not in ("succeeded", "failed", "canceled") and time.time() < deadline:
        time.sleep(3)
        r = requests.get(f"{REPLICATE_API}/predictions/{pred_id}", headers=h_get, timeout=30)
        pred = r.json()
        status = pred["status"]
    if status != "succeeded":
        raise RuntimeError(f"Replicate {status}: {pred.get('error') or pred.get('logs','')[:300]}")
    out = pred.get("output")
    if out is None:
        raise RuntimeError("No output")
    return [out] if isinstance(out, str) else list(out)


def _call_render(image: Image.Image, prompt: str, mode: str, steps: int = 30, extra: str = "") -> Image.Image:
    """根據 mode 分流到不同後端。"""
    full_prompt = prompt + (", " + extra.strip() if extra.strip() else "")

    if mode == MODE_SKETCH:
        # SDXL + Canny ControlNet (描邊)
        model_ver = SKETCH_MODEL.split(":", 1)[1]
        full_prompt = full_prompt + ", professional interior photography, soft lighting"
        out = _replicate_run(model_ver, {
            "image": _to_data_url(image),
            "prompt": full_prompt,
            "negative_prompt": NEG_PROMPT_SKETCH,
            "num_inference_steps": int(steps),
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 0.8,
        }, timeout=180)
    else:
        # Interior Design SDXL + Depth ControlNet (保結構)
        model_ver = PHOTO_MODEL.split(":", 1)[1]
        full_prompt = "masterfully designed interior, " + full_prompt
        out = _replicate_run(model_ver, {
            "image": _to_data_url(image),
            "prompt": full_prompt,
            "negative_prompt": NEG_PROMPT_PHOTO,
            "depth_strength": 0.85,
            "promax_strength": 0.85,
            "refiner_strength": 0.4,
            "guidance_scale": 7.5,
            "num_inference_steps": int(steps),
        }, timeout=300)

    url = out[0]
    return Image.open(io.BytesIO(requests.get(url, timeout=120).content)).convert("RGB")


# ===== Handlers =====
def render_one(image: Image.Image, mode: str, style_key: str, extra: str, steps: int):
    print(f"[render] render_one  mode={mode!r}  style={style_key}  steps={steps}", flush=True)
    if image is None:
        return None, "請先上傳圖片"
    if style_key not in STYLES:
        return None, f"未知風格:{style_key}"
    try:
        result = _call_render(image, STYLES[style_key], mode, steps=steps, extra=extra)
        return result, f"✓ 模式:**{mode}** · 風格:**{style_key}**"
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ 失敗:{type(e).__name__}: {e}"


def render_all(image: Image.Image, mode: str, extra: str, steps: int):
    """一鍵跑全 6 風格對比。"""
    print(f"[render] render_all  mode={mode!r}", flush=True)
    if image is None:
        return [], "請先上傳圖片"
    results = []
    log = []
    for style, prompt_template in STYLES.items():
        try:
            img = _call_render(image, prompt_template, mode, steps=steps, extra=extra)
            results.append((img, style))
            log.append(f"✓ {style}")
        except Exception as e:
            log.append(f"❌ {style}: {str(e)[:80]}")
    return results, "\n".join(log)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🎨",
            title="AI 室內設計師",
            subtitle="一張圖換 6 種風格 — 自動分流:線稿用 Canny ControlNet 從 0 創造,現場照片用 Depth ControlNet 保結構換家具",
            tools=[
                ("SDXL + Canny ControlNet", "線稿模式:lucataco/sdxl-controlnet,保留每條線"),
                ("Interior Design SDXL + Depth", "照片模式:rocketdigitalai/interior-design-sdxl,保 3D 結構"),
                ("Replicate API", "雲端 GPU 推論"),
            ],
            cost="$0.02-0.03 / 張",
            cost_detail="6 風格對比 ≈ $0.15",
            time="~10-15 秒 / 張",
            time_detail="6 風格平行 ≈ 1-2 分鐘",
            badges=["雙模式", "雲端推論", "6 風格"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:先告訴 AI 你的輸入是什麼類型(線稿 vs 現場照片),系統會自動切到最適合的模型。然後選風格 → 生成。要對比 6 種風格?點下面那顆「全 6 風格」按鈕。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                mode = gr.Radio(
                    label="你的輸入是什麼?",
                    choices=[MODE_SKETCH, MODE_PHOTO],
                    value=MODE_SKETCH,
                    info="不同輸入用不同 ControlNet:線稿描邊嚴格 · 照片保 3D 深度",
                )
                in_img = gr.Image(type="pil", label="上傳", height=320)
                style = gr.Radio(
                    label="想擺成哪種風格?",
                    choices=list(STYLES.keys()),
                    value="🇸🇪 北歐極簡",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn1 = gr.Button("🎨 生成單一風格", variant="primary", scale=2)
                with gr.Row(elem_classes=["demo-secondary"]):
                    btn_all = gr.Button("🎨🎨 跑全 6 風格對比 (~2 分鐘, $0.15)")
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    extra = gr.Textbox(
                        label="額外英文描述 (選填)",
                        placeholder="e.g. with large window, ocean view, golden hour light",
                        lines=2,
                        info="補充給 AI 的提示,可影響光線、視角、特定家具",
                    )
                    steps = gr.Slider(
                        20, 50, value=30, step=5,
                        label="生成精緻度",
                        info="越高越精緻越慢,預設 30 為品質/速度平衡點",
                    )
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 渲染結果", elem_classes=["demo-section-title"])
                single_out = gr.Image(type="pil", label=None, show_label=False, height=420)
                log_md = gr.Markdown()
                gr.Markdown("### 6 風格並排比較", elem_classes=["demo-section-title"])
                gallery_out = gr.Gallery(label=None, show_label=False, columns=3, height=420, object_fit="contain")

        btn1.click(render_one, [in_img, mode, style, extra, steps], [single_out, log_md])
        btn_all.click(render_all, [in_img, mode, extra, steps], [gallery_out, log_md])

    return demo
