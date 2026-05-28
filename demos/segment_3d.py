"""物件切割 → 3D mesh — Tier 2 + Tier 3 整合。

流程:
  1. 上傳室內照片
  2. 輸入要切的物件 (英文,e.g. "sofa") → Grounded SAM 自動找+切
  3. 顯示去背 cutout
  4. 點「Make 3D」→ InstantMesh 生 3D mesh
  5. gr.Model3D 嵌入瀏覽器,**可拖曳旋轉**

工具:
  - Grounded SAM (text→mask):schananas/grounded_sam
  - 3D mesh:camenduru/instantmesh
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import requests
from PIL import Image


REPLICATE_API = "https://api.replicate.com/v1"
SAM_VERSION = "ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"   # schananas/grounded_sam
MESH_VERSION = "4f151757fd04d508b84f2192a17f58d11673971f05d9cb1fd8bd8149c6fc7cbb"  # camenduru/instantmesh


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError(
            "REPLICATE_API_TOKEN 未設定,請編輯 .env 加入 https://replicate.com/account/api-tokens 拿的 token"
        )
    return t


def _to_data_url(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _replicate_run(version: str, inputs: dict, timeout: int = 600) -> list[str]:
    """跑 prediction 直到完成 — 非同步 create + poll。429 自動重試。"""
    token = _token()
    h_post = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    h_get = {"Authorization": f"Bearer {token}"}
    body = {"version": version, "input": inputs}

    # 1. 建 prediction (不要 Prefer:wait,避免 long-hold timeout)
    pred = None
    for attempt in range(6):
        r = requests.post(f"{REPLICATE_API}/predictions", headers=h_post, json=body, timeout=90)
        if r.status_code == 429:
            wait = int(r.json().get("retry_after", 6)) + 2
            print(f"[segment_3d] rate-limited, 等 {wait}s 重試 (attempt {attempt+1}/6)", flush=True)
            time.sleep(wait)
            continue
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Replicate API {r.status_code}: {r.text[:300]}")
        pred = r.json()
        break
    if pred is None:
        raise RuntimeError("Replicate API 持續 429,放棄")

    pred_id = pred["id"]
    status = pred["status"]
    print(f"[segment_3d] prediction {pred_id} created, polling ...", flush=True)

    # 2. Poll 直到完成
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
    if isinstance(out, str):
        return [out]
    return list(out)


def _download(url: str, ext: str = "") -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=ext or os.path.splitext(url.split("?")[0])[1])
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path


# ===== Step 1: segment =====
def segment(image: Image.Image, mask_prompt: str, negative: str):
    print(f"[segment_3d] segment  mask_prompt={mask_prompt!r}", flush=True)
    if image is None:
        return None, None, "請先上傳圖片"
    if not mask_prompt.strip():
        return None, None, "請填入要切的物件 (例如 sofa)"

    # 先把原圖縮到合理大小,後續本機處理用
    src = image.convert("RGB").copy()
    src.thumbnail((1024, 1024))

    try:
        outputs = _replicate_run(
            SAM_VERSION,
            {
                "image": _to_data_url(src),
                "mask_prompt": mask_prompt.strip(),
                "negative_mask_prompt": (negative or "").strip(),
                "adjustment_factor": 0,
            },
            timeout=180,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, f"❌ 切割失敗:{type(e).__name__}: {e}"

    print(f"[segment_3d] outputs: {outputs}", flush=True)
    # outputs 通常是 [annotated, neg_annotated, mask, inverted_mask]
    # 找到 mask.jpg (不是 inverted_mask 也不是 annotated)
    mask_url = None
    for u in outputs:
        low = u.lower()
        if "/mask." in low or low.endswith("/mask.jpg") or low.endswith("/mask.png"):
            if "inverted" not in low:
                mask_url = u
                break
    if mask_url is None:
        # fallback: 找含 'mask' 但不含 'inverted' 也不含 'annotated' 的
        for u in outputs:
            low = u.lower()
            if "mask" in low and "inverted" not in low and "annotated" not in low:
                mask_url = u
                break
    if mask_url is None:
        return None, None, f"❌ 從 SAM 輸出找不到 mask 檔。Outputs: {outputs}"

    # 下載 mask + 套到原圖
    try:
        mask = Image.open(io.BytesIO(requests.get(mask_url, timeout=60).content)).convert("L")
        # mask 跟 src 尺寸對齊
        if mask.size != src.size:
            mask = mask.resize(src.size, Image.LANCZOS)
        # 套用:用 mask 當 alpha channel
        cutout = src.copy()
        cutout.putalpha(mask)
        # 存暫存 PNG (含 alpha) 給下一步用
        fd, cutout_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cutout.save(cutout_path, format="PNG")
    except Exception as e:
        return None, None, f"❌ 套用 mask 失敗:{e}"

    return cutout, cutout_path, f"✓ 切出 `{mask_prompt}` (本機 cutout 已存),接著按「Make 3D」"


# ===== Step 2: 3D mesh =====
def make_3d(cutout_path: str, sample_steps: int):
    print(f"[segment_3d] make_3d  steps={sample_steps}  cutout_path={cutout_path}", flush=True)
    if not cutout_path or not os.path.exists(cutout_path):
        return None, "請先做切割步驟"

    # 把本機 PNG cutout 轉成 data URL 給 Replicate
    try:
        cutout_img = Image.open(cutout_path)
        # InstantMesh 接受白色背景的圖,我們 alpha=0 改成白色
        if cutout_img.mode == "RGBA":
            bg = Image.new("RGB", cutout_img.size, (255, 255, 255))
            bg.paste(cutout_img, mask=cutout_img.split()[-1])
            cutout_img = bg
        cutout_data_url = _to_data_url(cutout_img)
    except Exception as e:
        return None, f"❌ 讀取 cutout 失敗:{e}"

    try:
        outputs = _replicate_run(
            MESH_VERSION,
            {
                "image_path": cutout_data_url,
                "seed": 42,
                "sample_steps": int(sample_steps),
                "export_video": False,
                "export_texmap": False,
                "remove_background": False,  # 我們已經處理過了
            },
            timeout=600,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ 3D 生成失敗:{type(e).__name__}: {e}"

    print(f"[segment_3d] mesh outputs: {outputs}", flush=True)
    # 找 .obj / .glb / .ply 檔案
    mesh_url = None
    for u in outputs:
        low = u.lower().split("?")[0]
        if low.endswith((".obj", ".glb", ".ply")):
            mesh_url = u
            break
    if not mesh_url:
        mesh_url = outputs[0]  # 沒副檔名也試試

    try:
        ext = os.path.splitext(mesh_url.split("?")[0])[1] or ".obj"
        local_path = _download(mesh_url, ext=ext)
    except Exception as e:
        return None, f"❌ 下載 mesh 失敗:{e}\n\nURL 列表:{outputs}"

    return local_path, f"✓ 3D mesh 完成 (本機: `{local_path}`)\n\n用滑鼠**拖曳旋轉**、**滾輪縮放**。"


# ===== build =====
def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🎯",
            title="物件切割 → 可拖旋 3D",
            subtitle="從室內照片切出單一物件,用單張圖生成 3D mesh,在瀏覽器中可拖曳旋轉",
            tools=[
                ("Grounded SAM", "text-prompted segmentation (schananas/grounded_sam)"),
                ("InstantMesh", "單圖→3D mesh 重建 (camenduru/instantmesh)"),
                ("gr.Model3D / Three.js", "瀏覽器原生 WebGL 拖旋預覽"),
            ],
            cost="~$0.05-0.10",
            cost_detail="SAM $0.005 + InstantMesh $0.05-0.1",
            time="40-120 秒",
            time_detail="SAM ~5-10s + InstantMesh ~30-90s",
            badges=["雲端推論", "3D 輸出", "適合規則物件"],
        )
        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:兩步驟 — 第一步從照片切出單一物件,第二步把它變成可在瀏覽器拖曳旋轉的 3D 模型。<br/>📌 <strong>適合幾何規則的物件</strong>:椅子、桌子、燈具、花瓶。沙發/植物/整個房間效果會差。</div>')

        gr.Markdown("### 步驟 1 · 從照片切出物件", elem_classes=["demo-section-title"])
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="上傳室內照片 / 物件特寫", height=280)
                mask_prompt = gr.Textbox(
                    label="想切出哪個物件? (英文)",
                    placeholder="chair / lamp / vase / table",
                    value="chair",
                    info="建議用英文最準確 (DINO 為英文訓練模型)",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    seg_btn = gr.Button("✂️ 切出物件", variant="primary", scale=2)
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    negative = gr.Textbox(
                        label="排除的元素 (選填,英文)",
                        placeholder="floor, wall, ceiling",
                        value="",
                        info="想避免被一起切到的東西",
                    )
            with gr.Column(scale=1, elem_classes=["demo-output-pane"]):
                cutout = gr.Image(type="pil", label="去背後的物件", height=280)
                seg_log = gr.Markdown()

        gr.Markdown("### 步驟 2 · 轉成可拖旋 3D 模型", elem_classes=["demo-section-title"])
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                with gr.Row(elem_classes=["demo-cta"]):
                    mesh_btn = gr.Button("🧊 生成 3D 模型 (約 1 分鐘)", variant="primary", scale=2)
                mesh_log = gr.Markdown()
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    sample_steps = gr.Slider(30, 100, value=50, step=10, label="3D 精緻度",
                                             info="越高越精緻越慢。預設 50 平衡。")
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                mesh_view = gr.Model3D(label="🖱️ 拖曳旋轉 / 滾輪縮放 / Shift+拖曳平移", height=480)

        state_masked_url = gr.State("")

        seg_btn.click(segment, [in_img, mask_prompt, negative], [cutout, state_masked_url, seg_log])
        mesh_btn.click(make_3d, [state_masked_url, sample_steps], [mesh_view, mesh_log])

    return demo
