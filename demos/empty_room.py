"""空房效果 — 上傳照片自動移除家具/家電/裝飾,留下建築結構。

流程:
  1. Grounded SAM 偵測所有家具/家電/裝飾,取得 union mask
  2. 把 mask 做 dilation (擴張) 留邊保險
  3. SDXL Inpainting + empty-room prompt 填補
  4. before / after 對比

成本 ~ $0.04 / 次 (SAM $0.005 + SDXL inpaint $0.035)
時間 ~ 30-60 秒
"""

from __future__ import annotations

import base64
import io
import os
import time

import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageFilter


REPLICATE_API = "https://api.replicate.com/v1"
SAM_VERSION = "ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"            # schananas/grounded_sam
INPAINT_VERSION = "a5b13068cc81a89a4fbeefeccc774869fcb34df4dbc92c1555e0f2771d49dde7"        # lucataco/sdxl-inpainting

# 分組 prompt — 每組 ≤ 5 個項目,grounded_sam 對長 prompt 偶爾會 0 偵測爆 reshape error
# 多次跑 + 合併 mask,提升健壯性 (失敗 batch 自動跳過)
PROMPT_BATCHES = [
    "sofa, couch, chair, armchair, ottoman",      # 大型坐具
    "table, desk, coffee table, nightstand",      # 桌類
    "bed, mattress, headboard",                    # 床
    "lamp, chandelier, light fixture",             # 燈具
    "cabinet, shelf, bookshelf, wardrobe",         # 收納
    "tv, television, monitor, computer",           # 電器
    "rug, carpet, mat",                            # 地毯
    "cushion, pillow, blanket, throw",             # 軟裝
    "plant, flower, vase, pot",                    # 植物
    "painting, picture, mirror, art frame",        # 牆面藝術
    "clock, decoration, ornament, books",          # 小擺件
    "curtain, blind, drape",                       # 窗簾
]

NEGATIVE_MASK = "wall, floor, ceiling, window, door, frame, sky, outdoor"

EMPTY_ROOM_PROMPT = (
    "empty interior room, plain white walls, clean wooden floor, no furniture, "
    "no decoration, minimal architectural shell, soft natural light, photorealistic, 8k"
)

EMPTY_ROOM_NEG = (
    "furniture, sofa, chair, table, lamp, plant, painting, decoration, "
    "rug, cushion, art, low quality, blurry, distorted"
)


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError("REPLICATE_API_TOKEN 未設定")
    return t


def _to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.standard_b64encode(buf.getvalue()).decode()}"


def _replicate_run(version: str, inputs: dict, timeout: int = 600) -> list[str]:
    token = _token()
    h_post = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    h_get = {"Authorization": f"Bearer {token}"}
    body = {"version": version, "input": inputs}

    pred = None
    for attempt in range(6):
        r = requests.post(f"{REPLICATE_API}/predictions", headers=h_post, json=body, timeout=90)
        if r.status_code == 429:
            wait = int(r.json().get("retry_after", 6)) + 2
            print(f"[empty_room] rate-limited, 等 {wait}s 重試 ({attempt+1}/6)", flush=True)
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


def _find_mask_url(outputs: list[str]) -> str | None:
    for u in outputs:
        low = u.lower()
        if "inverted" in low or "annotated" in low:
            continue
        if "/mask." in low or low.endswith("/mask.jpg") or low.endswith("/mask.png"):
            return u
    # fallback
    for u in outputs:
        low = u.lower()
        if "mask" in low and "inverted" not in low and "annotated" not in low:
            return u
    return None


def empty_room(image: Image.Image, mask_dilate: int, inpaint_steps: int):
    print(f"[empty_room] start  dilate={mask_dilate}  steps={inpaint_steps}", flush=True)
    if image is None:
        return None, None, None, "請先上傳圖片"

    # 縮到 SDXL 友善尺寸 (1024 邊長,長寬皆為 8 的倍數)
    src = image.convert("RGB").copy()
    src.thumbnail((1024, 1024))
    w, h = src.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    src = src.resize((w, h), Image.LANCZOS)

    # ===== Step 1: 多次 Grounded SAM (短 prompt) 合併 mask =====
    src_data_url = _to_data_url(src, "PNG")
    combined_mask = Image.new("L", (w, h), 0)  # 全黑底,合併時用 OR
    used_passes = 0
    failed_passes = []

    for batch in PROMPT_BATCHES:
        try:
            sam_out = _replicate_run(
                SAM_VERSION,
                {
                    "image": src_data_url,
                    "mask_prompt": batch,
                    "negative_mask_prompt": NEGATIVE_MASK,
                    "adjustment_factor": 6,
                },
                timeout=180,
            )
        except Exception as e:
            # 該 batch 可能沒偵測到任何物件 (常見 0-tensor reshape error),跳過
            print(f"[empty_room] batch 跳過 ({batch[:30]}...): {str(e)[:80]}", flush=True)
            failed_passes.append(batch[:30])
            continue

        mask_url = _find_mask_url(sam_out)
        if not mask_url:
            continue
        try:
            m = Image.open(io.BytesIO(requests.get(mask_url, timeout=60).content)).convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.LANCZOS)
            # OR 合併 (取兩 mask 的最大值)
            combined = np.maximum(np.array(combined_mask), np.array(m))
            combined_mask = Image.fromarray(combined, mode="L")
            used_passes += 1
            print(f"[empty_room] batch ✓ ({batch[:30]}...)", flush=True)
        except Exception as e:
            print(f"[empty_room] 下載 mask 失敗 ({batch[:30]}...): {e}", flush=True)
            continue

    if used_passes == 0:
        return None, None, None, "❌ 全部 batch 都沒偵測到家具。可能這張圖已經是空房或物件太特殊。"

    print(f"[empty_room] {used_passes}/{len(PROMPT_BATCHES)} batches 成功", flush=True)
    mask_img = combined_mask

    # Dilation:把 mask 擴張一點,避免邊緣殘影
    if mask_dilate > 0:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(int(mask_dilate) * 2 + 1))

    # Mask 預覽圖 (把 mask 疊在原圖上方便檢視)
    overlay = src.copy().convert("RGBA")
    red = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
    mask_rgba = mask_img.convert("L").point(lambda x: 120 if x > 30 else 0)
    red.putalpha(mask_rgba)
    overlay.alpha_composite(red)
    mask_preview = overlay.convert("RGB")

    # ===== Step 2: SDXL inpainting =====
    try:
        inp_out = _replicate_run(
            INPAINT_VERSION,
            {
                "image": _to_data_url(src, "PNG"),
                "mask": _to_data_url(mask_img, "PNG"),
                "prompt": EMPTY_ROOM_PROMPT,
                "negative_prompt": EMPTY_ROOM_NEG,
                "steps": int(inpaint_steps),
                "guidance_scale": 8,
                "strength": 0.99,
                "num_outputs": 1,
            },
            timeout=300,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return src, mask_preview, None, f"❌ Inpainting 失敗:{e}\n\n(只能給你看 mask 預覽)"

    print(f"[empty_room] inpaint outputs: {inp_out}", flush=True)
    if not inp_out:
        return src, mask_preview, None, "❌ Inpainting 沒回傳"
    try:
        inpaint_raw = Image.open(io.BytesIO(requests.get(inp_out[0], timeout=120).content)).convert("RGB")
    except Exception as e:
        return src, mask_preview, None, f"❌ 下載結果失敗:{e}"

    # === 關鍵:合成 — 只在 mask 內保留 inpaint 結果,mask 外用原圖蓋回 ===
    # 避免 SDXL VAE 偷偷改寫 mask 外的牆/天花板/門窗
    if inpaint_raw.size != src.size:
        inpaint_raw = inpaint_raw.resize(src.size, Image.LANCZOS)

    # 把 mask 邊緣羽化 (Gaussian blur),讓接縫平滑不可見
    mask_feathered = mask_img.filter(ImageFilter.GaussianBlur(radius=6))
    mask_arr = np.asarray(mask_feathered).astype(np.float32) / 255.0
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[:, :, None]

    orig_arr = np.asarray(src).astype(np.float32)
    inpaint_arr = np.asarray(inpaint_raw).astype(np.float32)

    # alpha blend:mask=1 用 inpaint,mask=0 用原圖
    composed = orig_arr * (1.0 - mask_arr) + inpaint_arr * mask_arr
    result_img = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))

    log = f"✓ 完成 ({used_passes}/{len(PROMPT_BATCHES)} SAM batch 成功) · mask 外保留原圖"
    if failed_passes:
        log += f"\n\n_略過 batches: {', '.join(failed_passes)}_"
    return src, mask_preview, result_img, log


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🪄",
            title="一鍵空房 (移除所有家具)",
            subtitle="房地產 staging 神器 — AI 偵測並移除所有家具/家電/裝飾,還原成只有牆地天的建築結構",
            tools=[
                ("Grounded SAM (多 pass)", "12 組短 prompt 平行偵測 30+ 種家具,union 合併 mask"),
                ("SDXL Inpainting", "用 empty-room prompt 填補移除區域"),
                ("Mask 羽化合成", "Gaussian blur + alpha blend,確保 mask 外像素零改動"),
            ],
            cost="~$0.05-0.08",
            cost_detail="多次 SAM + 1 次 SDXL inpaint",
            time="60-200 秒",
            time_detail="低 credit 時受 burst=1 限速影響",
            badges=["房產 staging", "多 pass 容錯", "原圖保留"],
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳一張室內照片,AI 自動辨識並移除所有家具、家電、裝飾,變回只有牆/地板/門窗的「空房狀態」。適合房地產 staging、裝修前後對比、空間重新規劃。</div>')
                in_img = gr.Image(type="pil", label="上傳室內照片", height=300)
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🪄 變成空房 (約 1-2 分鐘)", variant="primary", scale=2)
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    dilate = gr.Slider(0, 15, value=5, step=1, label="邊緣擴張範圍",
                                       info="太小:家具邊緣會有殘影。太大:可能蓋到牆面。預設 5 適合多數情況。")
                    steps = gr.Slider(20, 50, value=30, step=5, label="生成精緻度",
                                      info="越高越精緻越慢。預設 30 平衡。")
                log_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 偵測過程", elem_classes=["demo-section-title"])
                with gr.Row():
                    orig_view = gr.Image(type="pil", label="原圖", height=320)
                    mask_view = gr.Image(type="pil", label="AI 識別的家具區域", height=320)
                gr.Markdown("### 空房結果", elem_classes=["demo-section-title"])
                result_view = gr.Image(type="pil", label="清空後的空間", height=380)

        btn.click(empty_room, [in_img, dilate, steps], [orig_view, mask_view, result_view, log_md])

    return demo
