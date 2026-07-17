# Orin 本機影片剪輯與摘要

這是一個 CLI-only 專案，專門在 NVIDIA Jetson AGX Orin 上完成三件事：

1. 壓縮影片中過長的空白。
2. 使用本機 GPU Whisper 產生英文逐字稿。
3. 使用本機 Qwen 產生英文與繁體中文詳細摘要。

原始影片不會被修改，逐字稿與摘要不會送到雲端。

## 使用方式

完整流程（剪輯、逐字稿、雙語摘要）：

```bash
cd /home/jetson/video_editor
./run_docker.sh process "video/example.mp4"
```

只剪輯影片，不抽取 ASR 音訊、不載入 Whisper、不呼叫 Qwen：

```bash
./run_docker.sh process "video/example.mp4" --edit-only
```

只建立剪輯計畫，不輸出影片：

```bash
./run_docker.sh plan "video/example.mp4"
```

從既有工作的逐字稿重新產生摘要：

```bash
./run_docker.sh summarize output/JOB_ID
```

重新執行相同工作可加入 `--force`。自訂參數請複製
`config.example.json`，再使用 `--config your-config.json`。

## 摘要格式

摘要固定為 detailed overview，不再區分 standard/detailed，也不再產生
Key Takeaways、Uncertainties、Sections 或 Action Items。

模型原始 JSON 只有：

```json
{
  "title_en": "...",
  "title_zh_tw": "...",
  "overview_en": ["..."],
  "overview_zh_tw": ["..."],
  "completion_marker": "complete"
}
```

Prompt 目標是每種語言 10–14 段、英文 1,200–1,600 words。發布前只檢查：

- JSON 與完成標記完整。
- 至少 8 組英中對齊段落。
- 英文至少 1,000 words。
- 英文與繁中欄位使用正確文字系統。

程式不會修補、改寫或人工加工模型文字。`summary.raw.txt` 保存本機模型原始
回應，`summary.metrics.json` 保存 SHA-256 與 `post_generation_content_modified=false`。

## 預設本機模型

- ASR：`faster-whisper-large-v3-turbo`，CUDA FP16。
- 摘要：`qwen3.6:27b`，Ollama non-thinking one-shot。
- Ollama URL 強制限制為 localhost/loopback。

## 輸出

完整流程會建立 `output/<影片名稱>-<來源指紋>-<設定指紋>/`：

- `edited.mp4`
- `edit_plan.json`
- `transcript.json`、`transcript.srt`、`transcript.md`
- `summary.en.md`、`summary.zh-TW.md`
- `summary.json`、`summary.raw.txt`、`summary.metrics.json`
- `manifest.json` 與 FFmpeg log

`--edit-only` 使用獨立工作 ID，只輸出剪輯相關檔案，不會和完整流程的快取混淆。

## 專案結構

```text
cli.py         CLI 入口
pipeline.py    串接剪輯、ASR 與摘要
media.py       FFmpeg probe、剪輯與音訊抽取
silence.py     靜音區間與剪輯計畫
asr.py         faster-whisper
transcript.py  逐字稿輸出與 one-shot 視窗
summary.py     Ollama、簡化品質檢查與 Markdown
config.py      設定驗證
io_utils.py    原子寫檔與工作指紋
```

建置與測試：

```bash
./scripts/build_image.sh
./run_docker.sh test
```
