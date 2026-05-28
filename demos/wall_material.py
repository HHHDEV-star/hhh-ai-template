"""🪨 牆面材質模擬器 — 上傳房間,換牆面材質看效果。

8 種預設材質:
  · 北歐白漆 / 文藝復古磚 / 工業水泥 / 木質紋理
  · 大理石 / 莫蘭迪色油漆 / 灰岩紋 / 壁紙幾何

技術:Interior Design SDXL + Depth ControlNet (結構保留)
  輸入: 房間照片
  Prompt: 「同樣的房間,牆面換成 X 材質」
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
STAGING_VERSION = "a3c091059a25590ce2d5ea13651fab63f447f21760e50c358d4b850e844f59ee"


MATERIALS = {
    "🤍 北歐純白漆":        "matte white painted walls, smooth finish, scandinavian style, soft natural light, photorealistic, 8k",
    "🧱 文藝復古紅磚":      "exposed red brick wall, vintage industrial style, warm lighting, photorealistic, 8k",
    "🏭 工業水泥":          "concrete textured walls, industrial loft style, raw cement finish, moody atmosphere, photorealistic, 8k",
    "🌳 溫潤木質板":        "wooden panel walls, oak wood grain, japandi style, warm cozy lighting, photorealistic, 8k",
    "💎 大理石紋":          "white marble walls with subtle veining, luxury interior, elegant lighting, photorealistic, 8k",
    "🎨 莫蘭迪藕色":        "matte muted dusty pink walls, modern minimalist, soft pastel mood, photorealistic, 8k",
    "🗿 灰岩石材":          "natural grey stone walls, slate texture, modern mountain lodge style, photorealistic, 8k",
    "🌸 復古幾何壁紙":      "vintage geometric wallpaper, mid-century modern pattern, warm interior, photorealistic, 8k",
}

NEG_PROMPT = "ugly, deformed, noisy, blurry, low quality, distorted, watermark, text, cartoon"


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError("REPLICATE_API_TOKEN 未設定")
    return t


def _to_data_url(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.standard_b64encode(buf.getvalue()).decode()}"


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
            print(f"[wall_material] rate-limited, 等 {wait}s ({attempt+1}/6)", flush=True)
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


def _apply_material(image: Image.Image, material_prompt: str, depth: float = 0.92) -> Image.Image:
    """高 depth_strength 保留所有家具/結構,只換牆面氛圍。"""
    full_prompt = "interior room, " + material_prompt + ", keep all furniture and layout unchanged"
    out = _replicate_run(STAGING_VERSION, {
        "image": _to_data_url(image),
        "prompt": full_prompt,
        "negative_prompt": NEG_PROMPT,
        "depth_strength": depth,
        "promax_strength": 0.9,
        "refiner_strength": 0.35,
        "guidance_scale": 7.0,
        "num_inference_steps": 35,
    }, timeout=300)
    return Image.open(io.BytesIO(requests.get(out[0], timeout=120).content)).convert("RGB")


def apply_one(image: Image.Image, material_key: str):
    print(f"[wall_material] apply_one  material={material_key}", flush=True)
    if image is None:
        return None, None, "請先上傳房間照片"
    if material_key not in MATERIALS:
        return None, None, f"未知材質:{material_key}"
    try:
        result = _apply_material(image, MATERIALS[material_key])
        return image, result, f"✓ 套用 **{material_key}**"
    except Exception as e:
        import traceback; traceback.print_exc()
        return image, None, f"❌ 失敗:{e}"


def apply_all(image: Image.Image):
    """跑全 8 種材質做對比。"""
    print(f"[wall_material] apply_all", flush=True)
    if image is None:
        return [], "請先上傳房間照片"
    results = []
    log = []
    for key, prompt in MATERIALS.items():
        try:
            r = _apply_material(image, prompt)
            results.append((r, key))
            log.append(f"✓ {key}")
        except Exception as e:
            log.append(f"❌ {key}: {str(e)[:60]}")
    return results, "\n".join(log)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🪨",
            title="牆面材質模擬器",
            subtitle="上傳你家照片,8 種材質一鍵換 — 油漆、磁磚、木板、大理石、壁紙都能試,實體買單前先看效果",
            tools=[
                ("Interior Design SDXL + Depth", "高 depth 強度保留家具/結構,只換牆面材質"),
                ("Replicate API", "雲端 GPU 推論"),
                ("8 種預設材質", "北歐白漆 / 紅磚 / 水泥 / 木質 / 大理石 / 莫蘭迪 / 灰岩 / 壁紙"),
            ],
            cost="$0.03 / 張",
            cost_detail="8 材質對比 ≈ $0.24",
            time="~15 秒 / 張",
            time_detail="8 材質約 2-3 分鐘",
            badges=["視覺對比", "實用工具", "雲端"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳家裡或想設計的空間照片,8 種牆面材質一鍵換 — 看哪個最對味。<br/>📌 解決「貼了壁紙才後悔」「漆完油漆覺得不對」的痛點。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="上傳房間照片", height=320)
                material = gr.Radio(
                    label="想試哪種材質?",
                    choices=list(MATERIALS.keys()),
                    value="🤍 北歐純白漆",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn1 = gr.Button("🪨 套用單一材質 (~15s)", variant="primary", scale=2)
                with gr.Row(elem_classes=["demo-secondary"]):
                    btn_all = gr.Button("🪨🪨 跑全 8 種材質對比 (~3 min, $0.24)")
                log_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### Before / After", elem_classes=["demo-section-title"])
                with gr.Row():
                    before_view = gr.Image(type="pil", label="原圖", height=320)
                    after_view = gr.Image(type="pil", label="新材質", height=320)
                gr.Markdown("### 8 種材質並排", elem_classes=["demo-section-title"])
                gallery = gr.Gallery(label=None, show_label=False, columns=4, height=520, object_fit="contain")

        btn1.click(apply_one, [in_img, material], [before_view, after_view, log_md])
        btn_all.click(apply_all, [in_img], [gallery, log_md])

    return demo
