"""📸 AI 數位攝影師 — 隨手拍變雜誌劇照。

用 Magnific 等級的 clarity-upscaler 對室內照片做專業攝影師等級的增強:
  - 透視/銳利化 + 動態光影 + HDR + 細節補強
  - 4 種預設「攝影風格」:雜誌、晨光、戲劇、房地產

技術:philz1337x/clarity-upscaler (Magnific clone) via Replicate
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
CLARITY_VERSION = "dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"  # philz1337x/clarity-upscaler


# 四種「攝影師風格」預設 — 不同 prompt + creativity / dynamic 組合
PHOTO_PRESETS = {
    "📷 雜誌劇照 (中性自然)": {
        "prompt": "professional magazine interior photography, perfect natural lighting, sharp detail, balanced exposure, architectural digest style, high resolution, masterpiece",
        "creativity": 0.4,
        "resemblance": 0.85,
        "dynamic": 6,
        "sharpen": 1,
    },
    "🌅 晨光柔美 (溫暖明亮)": {
        "prompt": "professional interior photography, warm golden hour morning light, soft natural glow, dreamy atmosphere, sharp detail, masterpiece, high resolution",
        "creativity": 0.5,
        "resemblance": 0.8,
        "dynamic": 7,
        "sharpen": 1,
    },
    "🎬 戲劇質感 (電影感)": {
        "prompt": "cinematic interior photography, dramatic moody lighting, deep shadows, rich contrast, film grain, atmospheric, masterpiece, hyperdetailed",
        "creativity": 0.55,
        "resemblance": 0.75,
        "dynamic": 8,
        "sharpen": 2,
    },
    "🏠 房地產廣告 (清晰明亮)": {
        "prompt": "real estate listing photography, bright clean lighting, wide angle, crisp sharp detail, vibrant colors, spacious feel, professional, masterpiece",
        "creativity": 0.35,
        "resemblance": 0.9,
        "dynamic": 5,
        "sharpen": 2,
    },
}


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError("REPLICATE_API_TOKEN 未設定")
    return t


def _to_data_url(img: Image.Image) -> str:
    img = img.copy()
    if max(img.size) > 1280:
        img.thumbnail((1280, 1280))
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
            print(f"[photo_enhancer] rate-limited, 等 {wait}s ({attempt+1}/6)", flush=True)
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


def _enhance_call(image: Image.Image, preset_cfg: dict, scale: int = 2) -> Image.Image:
    out = _replicate_run(CLARITY_VERSION, {
        "image": _to_data_url(image),
        "prompt": preset_cfg["prompt"],
        "negative_prompt": "(worst quality, low quality, normal quality:2), blurry, deformed, oversaturated, cartoon",
        "creativity": preset_cfg["creativity"],
        "resemblance": preset_cfg["resemblance"],
        "dynamic": preset_cfg["dynamic"],
        "sharpen": preset_cfg["sharpen"],
        "scale_factor": scale,
    }, timeout=600)
    url = out[0]
    return Image.open(io.BytesIO(requests.get(url, timeout=120).content)).convert("RGB")


def enhance_one(image: Image.Image, preset_key: str, scale: int):
    print(f"[photo_enhancer] enhance_one  preset={preset_key}  scale={scale}", flush=True)
    if image is None:
        return None, None, "請先上傳照片"
    if preset_key not in PHOTO_PRESETS:
        return None, None, f"未知 preset:{preset_key}"
    cfg = PHOTO_PRESETS[preset_key]
    try:
        enhanced = _enhance_call(image, cfg, scale=int(scale))
        return image, enhanced, f"✓ 套用 **{preset_key}** · 放大 {scale}x"
    except Exception as e:
        import traceback; traceback.print_exc()
        return image, None, f"❌ 失敗:{type(e).__name__}: {e}"


def enhance_all(image: Image.Image, scale: int):
    print(f"[photo_enhancer] enhance_all  scale={scale}", flush=True)
    if image is None:
        return [], "請先上傳照片"
    results = []
    log = []
    for key, cfg in PHOTO_PRESETS.items():
        try:
            enhanced = _enhance_call(image, cfg, scale=int(scale))
            results.append((enhanced, key))
            log.append(f"✓ {key}")
        except Exception as e:
            log.append(f"❌ {key}: {str(e)[:60]}")
    return results, "\n".join(log)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="📸",
            title="AI 數位攝影師",
            subtitle="把手機隨手拍的居家照變成雜誌封面等級劇照 — Magnific 等級的 HDR、銳利化、重打光,結構完全保留",
            tools=[
                ("Clarity Upscaler (Magnific clone)", "高解析度 + HDR + 細節補強,可調創意/相似度"),
                ("Replicate API", "雲端 GPU 推論 (philz1337x/clarity-upscaler)"),
                ("4 種攝影預設", "雜誌中性 / 晨光柔美 / 戲劇電影感 / 房地產廣告"),
            ],
            cost="$0.04 / 張",
            cost_detail="4 種對比 ≈ $0.16",
            time="~30-60 秒 / 張",
            time_detail="放大倍數越高越慢",
            badges=["攝影增強", "雲端", "HDR"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳手機拍的家裡照 (歪斜、光線雜、亂亂的也沒關係) → AI 重打光 + 補細節 + 加質感 → 變成雜誌劇照級。<br/>📌 用途:屋主社群分享、設計師交件、房屋出售廣告。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="上傳原始照片", height=320)
                preset = gr.Radio(
                    label="想要哪種攝影風格?",
                    choices=list(PHOTO_PRESETS.keys()),
                    value="📷 雜誌劇照 (中性自然)",
                )
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    scale = gr.Slider(1, 4, value=2, step=1, label="放大倍數",
                                      info="2x 推薦,4x 更精細但時間翻倍")
                with gr.Row(elem_classes=["demo-cta"]):
                    btn1 = gr.Button("📸 套用攝影師風格 (~30s)", variant="primary", scale=2)
                with gr.Row(elem_classes=["demo-secondary"]):
                    btn_all = gr.Button("📸📸 跑全 4 種風格 (~2 分鐘, $0.16)")
                log_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### Before / After", elem_classes=["demo-section-title"])
                with gr.Row():
                    before_view = gr.Image(type="pil", label="原圖", height=320)
                    after_view = gr.Image(type="pil", label="增強後", height=320)
                gr.Markdown("### 4 種攝影風格並排", elem_classes=["demo-section-title"])
                gallery = gr.Gallery(label=None, show_label=False, columns=2, height=480, object_fit="contain")

        btn1.click(enhance_one, [in_img, preset, scale], [before_view, after_view, log_md])
        btn_all.click(enhance_all, [in_img, scale], [gallery, log_md])

    return demo
