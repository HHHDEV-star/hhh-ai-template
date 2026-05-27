# hhh-ai-template

HHH 的 AI demo 集中地,用 [Gradio](https://gradio.app) 在本機跑。
每個 demo 一個檔案,放在 `demos/` 底下,啟動後在瀏覽器分頁切換。

## 啟動

```bash
# 第一次
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 之後每次
source .venv/bin/activate
python app.py
# → 自動打開 http://127.0.0.1:7860
```

## 加新 demo

1. 在 `demos/` 加一個 `<demo_name>.py`,export 一個 `build()` 函數回傳 `gr.Blocks` 或 `gr.Interface`
2. 在 `app.py` 的 `DEMOS` 加一行
3. 重啟 — 新 tab 就出現了

範本看 `demos/color_dna.py`(無 ML 依賴,純圖片處理,適合當 boilerplate)。

## 目前的 demo

| Tab | 檔案 | 說明 |
|---|---|---|
| 🎨 配色 DNA | `demos/color_dna.py` | 上傳圖,萃取主色調 5-7 色 |

之後規劃:
- 設計師風格指紋 (CLIP + 餘弦相似度)
- 以圖搜案例 (CLIP + Qdrant)
- 案例自動標籤 (CLIP zero-shot)
