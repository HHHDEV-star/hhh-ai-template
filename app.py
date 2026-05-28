"""HHH AI Lab — 室內設計 × 多模態 AI demo collection。

架構:
  - 頂部品牌化 header
  - 左側分類 sidebar nav (按類別群組)
  - 右側 demo 內容區 (Column visibility 切換,只顯示當前 demo)

加新 demo:在 `DEMOS_BY_CATEGORY` 加一個 entry,sidebar / 顯示區自動處理。
"""

from functools import partial

from dotenv import load_dotenv
load_dotenv()  # 必須在 import demos 之前,讓 ANTHROPIC_API_KEY / REPLICATE_API_TOKEN 等可讀

import gradio as gr

from demos import (
    alternatives, budget_therapist, case_seo, color_dna, dna_tinder,
    empty_room, floor_plan, full_plan, object_qa, pet_safety, photo_enhancer,
    render, segment_3d, style_fingerprint, style_map, sunlight, time_machine,
    trend_dashboard, walkthrough, wall_material, welcome,
)
from demos._ui import GLOBAL_CSS


# === Demo 分類 ===
DEMOS_BY_CATEGORY = [
    ("👋 開始", [
        ("🏠", "Welcome",                   welcome.build),
    ]),
    ("🌟 旗艦體驗 · Flagship", [
        ("🕰️", "我家 Time Machine", time_machine.build),
        ("🏗️", "3 分鐘完整翻新計畫",        full_plan.build),
    ]),
    ("互動式 AI · Interactive", [
        ("🧬", "風格 DNA Tinder",   dna_tinder.build),
        ("💸", "預算焦慮治療師",    budget_therapist.build),
        ("📐", "AI 平面圖生成器",    floor_plan.build),
        ("☀️", "自然光全天追蹤",    sunlight.build),
        ("🐕", "寵物友善設計檢測",  pet_safety.build),
        ("🔍", "物件偵測 + Q&A",    object_qa.build),
    ]),
    ("資料洞察 · Analytics", [
        ("🔮", "設計趨勢儀表板",  trend_dashboard.build),
        ("👤", "設計師風格指紋",  style_fingerprint.build),
        ("🗺️", "設計師風格地圖",  style_map.build),
        ("🔁", "找類似案例",      alternatives.build),
    ]),
    ("內容生成 · Generative", [
        ("🎨", "AI 室內設計師",    render.build),
        ("📸", "AI 數位攝影師",    photo_enhancer.build),
        ("🪨", "牆面材質模擬器",    wall_material.build),
        ("🚶", "案例沉浸漫遊",     walkthrough.build),
        ("✍️", "AI 案例文案/SEO",  case_seo.build),
        ("🪄", "一鍵空房",         empty_room.build),
        ("🎯", "切割 → 3D",        segment_3d.build),
    ]),
    ("基礎工具 · Utility", [
        ("🌈", "配色 DNA", color_dna.build),
    ]),
]


THEME = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas"],
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0f172a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_radius="10px",
    button_primary_background_fill="#0f172a",
    button_primary_background_fill_hover="#1e293b",
    button_primary_text_color="#ffffff",
)


def header_html(total_demos: int) -> str:
    return f"""
<div class="app-header">
  <div class="brand">
    <div class="logo-mark">H</div>
    <div class="brand-text">
      <h1>HHH AI Lab</h1>
      <div class="tagline">室內設計 × 多模態 AI Demo Stack</div>
    </div>
  </div>
  <div class="header-meta">
    <div class="meta-pill"><span class="dot"></span>{total_demos} 個 demo 在線</div>
    <div class="meta-pill">CLIP · Claude · SDXL · DINO · SAM · InstantMesh</div>
  </div>
</div>
"""


def main() -> None:
    # 拍平所有 demo,維持分類順序
    flat: list[tuple[str, str, callable]] = []
    for _, items in DEMOS_BY_CATEGORY:
        flat.extend(items)

    with gr.Blocks(theme=THEME, css=GLOBAL_CSS, title="HHH AI Lab") as app:
        gr.HTML(header_html(len(flat)), elem_classes=["demo-hero-html"])

        with gr.Row():
            # === 側邊欄 ===
            with gr.Column(scale=1, min_width=220, elem_classes=["sidebar-col"]):
                with gr.Column(elem_classes=["sidebar-nav"]):
                    nav_buttons: list[gr.Button] = []
                    flat_index_map: dict[int, int] = {}  # button-list 內的 index → flat 中的 index
                    btn_idx = 0
                    flat_idx = 0
                    for cat_name, items in DEMOS_BY_CATEGORY:
                        gr.HTML(f'<div class="cat-label">{cat_name}</div>')
                        for icon, label, _ in items:
                            btn = gr.Button(
                                f"{icon}  {label}",
                                elem_classes=["nav-btn", "active" if flat_idx == 0 else ""],
                            )
                            nav_buttons.append(btn)
                            flat_index_map[btn_idx] = flat_idx
                            btn_idx += 1
                            flat_idx += 1

            # === 主內容區 ===
            with gr.Column(scale=5):
                demo_columns: list[gr.Column] = []
                for i, (_, _, build_fn) in enumerate(flat):
                    with gr.Column(visible=(i == 0)) as col:
                        build_fn()
                    demo_columns.append(col)

        # === 串接 sidebar 按鈕 → 切換 demo 可見性 + 更新 active class ===
        def make_switcher(target_flat_idx: int):
            def _switch():
                col_updates = [gr.update(visible=(i == target_flat_idx)) for i in range(len(demo_columns))]
                btn_updates = [
                    gr.update(elem_classes=["nav-btn", "active" if flat_index_map[i] == target_flat_idx else ""])
                    for i in range(len(nav_buttons))
                ]
                return col_updates + btn_updates
            return _switch

        for i, btn in enumerate(nav_buttons):
            btn.click(make_switcher(flat_index_map[i]), outputs=[*demo_columns, *nav_buttons])

    app.launch(inbrowser=True, server_name="127.0.0.1")


if __name__ == "__main__":
    main()
