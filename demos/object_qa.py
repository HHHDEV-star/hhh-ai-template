"""物件偵測 + 延伸提問 — Grounding DINO 找物件,Claude 生問題。

用法:設定 ANTHROPIC_API_KEY 進 .env,然後跑 app.py。
模型 ~340MB,第一次會自動下載。
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ===== 延遲載入 =====
_DINO_MODEL = None
_DINO_PROC = None
_CLAUDE = None

# 預設要偵測的物件類別 (DINO 需要英文,內部用)
DEFAULT_QUERY = (
    "sofa. lamp. chair. table. plant. curtain. cushion. rug. art. "
    "window. tv. door. shelf. bed. mirror. clock. vase. book."
)

# 英 → 中 對照表 — DINO 回傳英文,UI 顯示中文
LABEL_ZH = {
    "sofa": "沙發", "couch": "沙發", "armchair": "單椅", "chair": "椅子",
    "ottoman": "腳凳", "stool": "凳子", "bench": "長凳",
    "table": "桌子", "desk": "書桌", "coffee table": "茶几",
    "dining table": "餐桌", "nightstand": "床頭櫃", "side table": "邊桌",
    "bed": "床", "mattress": "床墊", "headboard": "床頭板",
    "lamp": "燈具", "light fixture": "燈具", "light": "燈具",
    "chandelier": "吊燈", "pendant light": "吊燈", "floor lamp": "落地燈",
    "table lamp": "桌燈", "wall light": "壁燈",
    "cabinet": "櫃子", "drawer": "抽屜櫃", "dresser": "斗櫃",
    "shelf": "層架", "bookshelf": "書架", "wardrobe": "衣櫃",
    "sideboard": "餐邊櫃", "storage": "收納櫃",
    "tv": "電視", "television": "電視", "monitor": "螢幕",
    "computer": "電腦", "speaker": "音響",
    "rug": "地毯", "carpet": "地毯", "mat": "踏墊",
    "cushion": "抱枕", "pillow": "枕頭", "blanket": "毯子", "throw": "披毯",
    "plant": "植栽", "flower": "花", "vase": "花瓶", "pot": "盆栽",
    "painting": "畫作", "picture": "畫作", "photo": "相片",
    "art": "藝術品", "art frame": "畫框", "mirror": "鏡子", "clock": "時鐘",
    "ornament": "擺飾", "decoration": "裝飾", "books": "書本", "magazine": "雜誌",
    "appliance": "家電", "refrigerator": "冰箱", "microwave": "微波爐",
    "oven": "烤箱", "dishwasher": "洗碗機",
    "curtain": "窗簾", "blind": "百葉窗", "drape": "窗簾",
    "window": "窗戶", "door": "門", "wall": "牆", "floor": "地板", "ceiling": "天花板",
    "fireplace": "壁爐", "fan": "風扇", "air conditioner": "冷氣",
}


def _to_zh(label: str) -> str:
    """英文 label → 中文。找不到就直接回原字串。"""
    key = (label or "").lower().strip()
    if key in LABEL_ZH:
        return LABEL_ZH[key]
    for prefix in ("a ", "an ", "the "):
        if key.startswith(prefix):
            sub = key[len(prefix):]
            if sub in LABEL_ZH:
                return LABEL_ZH[sub]
    return label


# 中 → 英 對照(從 LABEL_ZH 反推,同中文取「第一個英文」)
LABEL_EN = {}
for _en, _zh in LABEL_ZH.items():
    LABEL_EN.setdefault(_zh, _en)


# 預設的中文查詢清單 (UI 顯示給使用者)
DEFAULT_QUERY_ZH = "沙發、燈具、椅子、桌子、植栽、窗簾、抱枕、地毯、藝術品、窗戶、電視、門、層架、床、鏡子、時鐘、花瓶、書本"


def _normalize_query(raw: str) -> str:
    """把使用者輸入(中/英/混合,逗號、頓號、空格分隔)轉成 DINO 接受的英文 prompt。

    例:「沙發、椅子、lamp」 → "sofa. chair. lamp."
    """
    import re
    if not raw or not raw.strip():
        return ""
    # 統一分隔符:中文逗號、頓號、英文逗號、句號、分號 → 統一切
    parts = re.split(r"[,、,;.;]+", raw)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 先試中文 → 英文
        if p in LABEL_EN:
            out.append(LABEL_EN[p])
        else:
            # 直接當英文用(可能是 user 自訂的英文 prompt)
            out.append(p.lower())
    # DINO 用 ". " 分隔多 prompt
    return ". ".join(out) + "."

# 隨機色板給每個 box 不同顏色
PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b"]


def _load_dino():
    global _DINO_MODEL, _DINO_PROC
    if _DINO_MODEL is not None:
        return _DINO_MODEL, _DINO_PROC
    import torch  # noqa: F401
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print("[object_qa] loading Grounding DINO tiny ...", flush=True)
    model_id = "IDEA-Research/grounding-dino-tiny"
    _DINO_PROC = AutoProcessor.from_pretrained(model_id)
    _DINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    print("[object_qa] DINO ready", flush=True)
    return _DINO_MODEL, _DINO_PROC


def _claude():
    global _CLAUDE
    if _CLAUDE is not None:
        return _CLAUDE
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or "請貼" in key or key.startswith("sk-ant-api03-xxx"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY 未設定。請編輯 ~/github/hhh-ai-template/.env,"
            "把 placeholder 換成 https://console.anthropic.com/settings/keys 拿到的 key"
        )
    _CLAUDE = Anthropic(api_key=key)
    return _CLAUDE


# ===== 偵測 =====
def detect(image: Image.Image, query: str, threshold: float = 0.3, top_k: int = 10):
    import torch
    model, proc = _load_dino()
    image = image.convert("RGB")
    # 過大的圖先縮 (DINO 最大邊 1600 足夠)
    if max(image.size) > 1600:
        ratio = 1600 / max(image.size)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.LANCZOS)

    # 把中文/混合輸入轉成英文給 DINO
    en_query = _normalize_query(query) if query else ""
    if not en_query:
        en_query = "object."  # fallback
    print(f"[object_qa] detect query (zh): {query!r}  →  (en): {en_query!r}", flush=True)
    inputs = proc(images=image, text=en_query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=threshold, text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )[0]

    text_labels = results.get("text_labels", results.get("labels", []))
    objs = []
    for label, score, box in zip(text_labels, results["scores"], results["boxes"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        en = str(label)
        objs.append({
            "name": _to_zh(en),  # 顯示用 (中文)
            "name_en": en,       # 給 Claude prompt 用 (英文,精準)
            "score": float(score),
            "box": [x1, y1, x2, y2],
        })
    # 排序取信心度前 top_k,並去掉太小的框
    objs = [o for o in objs if (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) > 1500]
    objs = sorted(objs, key=lambda o: -o["score"])[:top_k]
    return image, objs


def draw_boxes(image: Image.Image, objects: list[dict]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, o in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = o["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # label 背景條
        tag = f"#{i+1} {o['name']} ({o['score']*100:.0f}%)"
        bbox = draw.textbbox((x1, y1), tag, font=font)
        pad = 4
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=color)
        draw.text((x1, y1), tag, fill="white", font=font)
    return out


# ===== Claude 生問題 =====
def ask_claude_for_questions(image: Image.Image, objects: list[dict]) -> dict[int, list[str]]:
    if not objects:
        return {}
    # 縮圖到合理大小再 base64 (Claude 限制 + 省錢)
    img = image.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    object_list = "\n".join(f"{i+1}. {o['name']}" for i, o in enumerate(objects))
    prompt = f"""這是一張室內裝修案例照片。物件偵測模型框出以下物件 (附編號):

{object_list}

請你扮演**居家裝潢顧問**,針對每個編號物件,從「**正準備裝修自己家的屋主**」角度,生成 **3 個** (剛好 3 個,不多不少) 居家裝潢相關的延伸提問。

問題必須:
- **看著照片講話**,不能是「沙發是什麼?」這種泛泛問題
- 聚焦在**屋主真的想知道**的事:預算/價位、哪裡買、CP 值替代、搭配什麼風格、適合多大坪數、有沒有平價款、材質保養、日常實用性、影響清潔/通風/動線等等
- 用**口語**寫,像屋主自己會問的話 (例:「這款沙發看起來像進口的,有沒有比較便宜的國產替代品?」)
- 每個問題針對該物件**該物件本身**,不要跨物件比較

範例 (假設第 1 個物件是「米白色 L 型布沙發」):
{{
  "1": [
    "這款米白布沙發看起來很好坐,大概什麼價位?有沒有 3 萬以內的類似款?",
    "米白色容易顯髒,有小孩或寵物的家庭該怎麼選材質才好清潔?",
    "L 型沙發要多大坪數客廳放才不會壓迫?我家 4 坪客廳適合嗎?"
  ]
}}

只回傳純 JSON,不要 markdown code fence,**所有 {len(objects)} 個編號都要有恰好 3 題**:
{{
  "1": ["...", "...", "..."],
  "2": ["...", "...", "..."],
  ...
}}"""

    client = _claude()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = msg.content[0].text.strip()
    print(f"[object_qa] Claude raw response (前 400 字):\n{text[:400]}\n...", flush=True)
    # 去掉 markdown fence 萬一有
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    data = json.loads(text)
    print(f"[object_qa] 解析後 keys={list(data.keys())[:5]}  第一個 value type={type(data[list(data.keys())[0]]).__name__}", flush=True)
    # key 可能是 str,轉成 int;value 必須是 list of strings
    result = {}
    for k, v in data.items():
        if isinstance(v, list):
            result[int(k)] = [str(x) for x in v]
        elif isinstance(v, str):
            result[int(k)] = [v]  # 萬一 Claude 給單字串,wrap 成 list
        else:
            result[int(k)] = []
    return result


# ===== Gradio handlers =====
def _format_questions(obj: dict) -> str:
    return f"### {obj['name']} (AI 信心度 {obj['score']*100:.0f}%)\n\n挑一個你最想知道的問題,按下方按鈕取得專業建議 👇"


def _ask_claude_for_answer(image: Image.Image, object_name: str, question: str) -> str:
    """看著照片回答關於某個物件的具體問題。"""
    img = image.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    prompt = f"""這是一張室內裝修案例照片。

屋主針對照片裡的「**{object_name}**」這個物件,問了你這個問題:

「{question}」

請以**居家裝潢顧問**身份,看著照片給屋主一個有用、具體、口語的回答。

要求:
- 回答**直接針對照片裡的這個物件**,不要泛泛而談
- 給**具體數字** (價位、尺寸、預算範圍) 如果合適
- 給**實用建議** (台灣可以買哪個品牌、可替代款、注意事項)
- 控制在 100-200 字,**繁體中文**
- 用 markdown 格式,可以用清單或粗體強調重點
"""
    client = _claude()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return msg.content[0].text.strip()


def analyze(image: Image.Image, query: str, threshold: float):
    print(f"[object_qa] analyze called  image={'None' if image is None else image.size}  threshold={threshold}", flush=True)
    empty = (None, [], None, "請先上傳圖片",
             gr.update(choices=[], value=None), "",
             gr.update(choices=[], value=None), "")
    if image is None:
        return empty
    try:
        resized, objs = detect(image, query, threshold=threshold, top_k=10)
        print(f"[object_qa] detect ok, {len(objs)} objects", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        return (None, [], None, f"❌ 偵測失敗:{type(e).__name__}: {e}",
                gr.update(choices=[], value=None), "",
                gr.update(choices=[], value=None), "")

    if not objs:
        return (resized, [], resized, "沒偵測到任何物件 — 試試降低 threshold 或修改要偵測的詞",
                gr.update(choices=[], value=None), "",
                gr.update(choices=[], value=None), "")

    annotated = draw_boxes(resized, objs)

    try:
        print("[object_qa] calling Claude ...", flush=True)
        questions = ask_claude_for_questions(resized, objs)
        print(f"[object_qa] Claude ok, got {sum(len(v) for v in questions.values())} questions", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        return (annotated, objs, resized, f"⚠ 偵測 OK,但 Claude 生問題失敗:{type(e).__name__}: {e}",
                gr.update(choices=[], value=None), "",
                gr.update(choices=[], value=None), "")

    for i, o in enumerate(objs, start=1):
        o["questions"] = questions.get(i, [])

    summary = f"偵測到 **{len(objs)}** 個物件:" + ", ".join(
        f"#{i+1} {o['name']}" for i, o in enumerate(objs)
    )
    obj_choices = [(f"#{i+1} {o['name']} ({o['score']*100:.0f}%)", i) for i, o in enumerate(objs)]
    first_q = _format_questions(objs[0]) if objs else ""
    first_q_choices = objs[0].get("questions", []) if objs else []
    return (annotated, objs, resized, summary,
            gr.update(choices=obj_choices, value=0), first_q,
            gr.update(choices=first_q_choices, value=None), "")


def show_questions(idx, objs):
    if idx is None or not objs or idx >= len(objs):
        return "", gr.update(choices=[], value=None), ""
    o = objs[idx]
    return _format_questions(o), gr.update(choices=o.get("questions", []), value=None), ""


def answer_question(question: str, idx, objs, resized_img: Image.Image):
    print(f"[object_qa] answer requested  obj_idx={idx}  q={(question or '')[:50]}", flush=True)
    if not question:
        return "請先選一個問題"
    if idx is None or not objs or idx >= len(objs):
        return "請先選一個物件"
    if resized_img is None:
        return "圖片資料遺失,請重新上傳分析"
    obj_name = objs[idx]["name"]
    try:
        answer = _ask_claude_for_answer(resized_img, obj_name, question)
        return f"**Q:** {question}\n\n---\n\n{answer}"
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ Claude 失敗:{type(e).__name__}: {e}"


# ===== build =====
def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🔍",
            title="物件偵測 + 智能 Q&A",
            subtitle="一張案例照變成可問可答的互動體驗 — DINO 框物件、Claude 看圖生問題+回答",
            tools=[
                ("Grounding DINO Tiny", "open-vocabulary 物件偵測,用文字 prompt 找任意家具"),
                ("Claude Sonnet 4.6 (vision)", "看圖生問題、看圖回答,中文裝潢顧問語氣"),
                ("Apple MPS", "DINO 本機 GPU 推論"),
            ],
            cost="$0.02-0.05",
            cost_detail="Claude vision API,一次分析 + 多次回答",
            time="5-10 秒 / 互動",
            time_detail="DINO ~2s · Claude ~3-5s",
            badges=["互動式 Q&A", "Claude API"],
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳一張你家或喜歡的居家照片,AI 會自動標出沙發、燈具、植栽等元素 — 點任一個物件,就會看到專屬於它的延伸提問與專業解答。</div>')
                in_img = gr.Image(type="pil", label="上傳室內照片", height=320)
                query = gr.Textbox(
                    value=DEFAULT_QUERY_ZH,
                    label="要偵測哪些元素?",
                    lines=2,
                    placeholder="例:沙發、椅子、燈具、窗簾、植栽",
                    info="用頓號或逗號分隔。中英皆可,可自由增減。",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🔍 開始分析", variant="primary", scale=2)
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    threshold = gr.Slider(
                        0.15, 0.6, value=0.3, step=0.05,
                        label="偵測敏感度",
                        info="越高越保守只抓最確定的;越低越貪心多抓但雜訊多。預設 0.3 適合多數情況。",
                    )
                summary_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                out_img = gr.Image(type="pil", label="AI 標記結果", height=380, show_label=True)
                gr.Markdown("### 互動 Q&A", elem_classes=["demo-section-title"])
                picker = gr.Dropdown(label="想了解哪個物件?", choices=[], value=None, interactive=True)
                q_md = gr.Markdown()
                q_radio = gr.Radio(label="挑一個問題", choices=[], value=None, interactive=True)
                with gr.Row(elem_classes=["demo-cta"]):
                    ask_btn = gr.Button("✨ 取得專業解答", variant="primary")
                answer_md = gr.Markdown()

        state_objs = gr.State([])
        state_resized_img = gr.State(None)  # 保留 resize 後的圖,供 Claude 回答時用

        btn.click(
            analyze,
            [in_img, query, threshold],
            [out_img, state_objs, state_resized_img, summary_md, picker, q_md, q_radio, answer_md],
        )
        picker.change(show_questions, [picker, state_objs], [q_md, q_radio, answer_md])
        ask_btn.click(answer_question, [q_radio, picker, state_objs, state_resized_img], answer_md)

    return demo
