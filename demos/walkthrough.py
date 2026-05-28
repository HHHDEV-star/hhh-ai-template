"""🚶 沉浸漫遊 — 從一張案例照生成「走進去」的短片漫遊。

流程:
  1. 上傳案例圖 或 從 hhh 資料庫 hcase_id 撈
  2. SVD (Stable Video Diffusion) 生成 14 frames 的 2-3 秒漫遊短片
  3. 同時讓 Claude 寫一段「設計師導覽」放在影片旁邊
  4. gr.Video 嵌入瀏覽器播放

預期效果:案例圖會出現緩慢推進 / 視角微移的動感,像「設計師帶你走入這個案例」。
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time

import gradio as gr
import requests
from PIL import Image


REPLICATE_API = "https://api.replicate.com/v1"
SVD_VERSION = "3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"  # stability-ai/stable-video-diffusion


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError("REPLICATE_API_TOKEN 未設定")
    return t


def _to_data_url(img: Image.Image) -> str:
    img = img.copy()
    # SVD 偏好 1024x576 或 576x1024
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return f"data:image/jpeg;base64,{base64.standard_b64encode(buf.getvalue()).decode()}"


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
            print(f"[walkthrough] rate-limited, 等 {wait}s 重試 ({attempt+1}/6)", flush=True)
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
    print(f"[walkthrough] prediction {pred_id} created, polling ...", flush=True)
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


def _claude_commentary(image: Image.Image) -> str:
    """請 Claude 看圖生一段「設計師導覽口吻」短文。"""
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return ""
    img = image.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    try:
        client = Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": """你是台灣資深室內設計師,正在帶屋主參觀這個你設計的案例。
請用第一人稱口吻 (我們),寫一段 100-150 字的導覽詞,讓人感覺真的有設計師在旁邊解說。
要點到:
- 整體風格定位
- 一個明顯的設計手法 (材質、配色、燈光、動線之一)
- 設計這樣做的「為什麼」

格式:純文字,2-3 句話,溫暖專業口吻,繁體中文。"""},
                ],
            }],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"[walkthrough] commentary 失敗 (不影響主流程): {e}", flush=True)
        return ""


def _fetch_from_hcase(hcase_id: int) -> Image.Image | None:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from db_utils import connect
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name FROM _hcase_img WHERE hcase_id = %s AND is_cover = 1 LIMIT 1",
                (hcase_id,),
            )
            r = cur.fetchone()
            if not r:
                cur.execute(
                    "SELECT name FROM _hcase_img WHERE hcase_id = %s ORDER BY hcase_img_id LIMIT 1",
                    (hcase_id,),
                )
                r = cur.fetchone()
        conn.close()
        if r and r["name"].startswith("http"):
            return Image.open(io.BytesIO(requests.get(r["name"], timeout=15).content)).convert("RGB")
    except Exception as e:
        print(f"[walkthrough] DB 撈圖失敗: {e}", flush=True)
    return None


def walkthrough(image: Image.Image, hcase_id_str: str, motion: int):
    print(f"[walkthrough] start  motion={motion}  hcase_id={hcase_id_str!r}", flush=True)
    # 圖片來源優先序:hcase_id > 上傳
    if hcase_id_str and hcase_id_str.strip().isdigit():
        img = _fetch_from_hcase(int(hcase_id_str.strip()))
        if img is None:
            return None, None, f"❌ 找不到 hcase_id={hcase_id_str.strip()} 的圖,改成上傳一張試試"
        image = img
    if image is None:
        return None, None, "請上傳案例圖,或輸入 hcase_id"

    # 1. SVD 生影片
    try:
        out = _replicate_run(SVD_VERSION, {
            "input_image": _to_data_url(image),
            "video_length": "14_frames_with_svd",
            "sizing_strategy": "maintain_aspect_ratio",
            "frames_per_second": 6,
            "motion_bucket_id": int(motion),
            "cond_aug": 0.02,
        }, timeout=600)
    except Exception as e:
        return None, None, f"❌ 影片生成失敗:{e}"

    video_url = out[0]
    print(f"[walkthrough] video URL: {video_url}", flush=True)
    # 下載 mp4 到 local 給 gr.Video
    try:
        r = requests.get(video_url, timeout=120)
        fd, path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
    except Exception as e:
        return None, None, f"❌ 下載影片失敗:{e}"

    # 2. Claude commentary (失敗不影響主流程)
    commentary = _claude_commentary(image)
    commentary_md = f"### 🎙️ 設計師導覽\n\n{commentary}" if commentary else ""

    return path, image, f"✓ 影片完成。motion={motion} 越高越動感。\n\n{commentary_md}"


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🚶",
            title="案例沉浸漫遊",
            subtitle="把靜態案例照變成「走進去」的動態漫遊短片 + 設計師導覽 — 從看圖到體驗的進化",
            tools=[
                ("Stable Video Diffusion", "從單張案例圖生成 14 frames 動態漫遊"),
                ("Claude Sonnet 4.6 (vision)", "看圖即時生成設計師導覽詞"),
                ("xoops _hcase_img", "可選從案例 ID 直接撈圖"),
            ],
            cost="~$0.15",
            cost_detail="主要是 SVD 模型推論",
            time="30-90 秒",
            time_detail="影片生成是主要耗時",
            badges=["影片生成", "沉浸體驗", "設計師導覽"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳任何案例圖,AI 自動生成短片漫遊 — 像「設計師帶你走進這個案例」的感覺。Claude 同時看圖生一段設計理念導覽,放在影片旁邊。<br/>📌 也可直接輸入 hhh 案例 ID 撈圖。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                with gr.Tabs():
                    with gr.Tab("📷 上傳案例圖"):
                        in_img = gr.Image(type="pil", label="案例照片", height=300)
                    with gr.Tab("🔖 從 hhh 案例庫"):
                        hcase_id = gr.Textbox(
                            label="hcase_id",
                            placeholder="例如:3",
                            info="輸入後會自動從 hhh DB 撈這個案例的圖",
                        )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🚶 生成漫遊短片", variant="primary", scale=2)
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    motion = gr.Slider(80, 200, value=127, step=1, label="動態強度",
                                       info="預設 127 適合室內。越高鏡頭移動越大但容易扭曲。")
                status_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 漫遊短片", elem_classes=["demo-section-title"])
                video_view = gr.Video(label=None, show_label=False, height=420, autoplay=True, loop=True)
                ref_view = gr.Image(label="原始圖", type="pil", height=200, interactive=False)

        btn.click(walkthrough, [in_img, hcase_id, motion], [video_view, ref_view, status_md])

    return demo
