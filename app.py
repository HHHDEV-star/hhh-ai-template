"""HHH AI Demo 集合入口。

每個 demo 一個檔案放 demos/,export `build()` 函數回傳 Gradio component。
在下方 DEMOS 加一行就會出現在 tab 列。
"""

import gradio as gr

from demos import color_dna, style_fingerprint

# (顯示名, build 函數) — 在這裡加新 demo
DEMOS = [
    ("👤 設計師風格指紋", style_fingerprint.build),
    ("🎨 配色 DNA", color_dna.build),
]


def main() -> None:
    blocks = []
    labels = []
    for label, build in DEMOS:
        blocks.append(build())
        labels.append(label)

    app = gr.TabbedInterface(
        blocks,
        labels,
        title="HHH AI Demos",
    )
    app.launch(inbrowser=True, server_name="127.0.0.1")


if __name__ == "__main__":
    main()
