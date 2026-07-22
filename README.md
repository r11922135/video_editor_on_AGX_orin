# Orin 本機影片剪輯與摘要

這是一個 CLI-only 專案，專門在 NVIDIA Jetson AGX Orin 上完成五件事：

1. 壓縮影片中過長的空白。
2. 使用本機 GPU Whisper 產生英文逐字稿。
3. 以固定規則把 ASR 片段整理成好讀的 Markdown 逐字稿。
4. 使用本機 Qwen 先產生英文詳細摘要，再另一次翻譯為台灣繁中。
5. 選用本機 Qwen 建立可稽核的術語校正規則、對時並燒錄英文字幕。

原始影片不會被修改，逐字稿與摘要不會送到雲端。

## 使用方式

完整流程（剪輯、逐字稿、雙語摘要）：

```bash
cd /home/jetson/video_editor
./run_docker.sh process "video/example.mp4"
```

完整流程並另外輸出燒錄字幕版影片：

```bash
./run_docker.sh process "video/example.mp4" --subtitles
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

只重新排版既有工作的 `transcript.md`，不重跑 ASR 或 LLM：

```bash
./run_docker.sh transcript output/JOB_ID
```

`process` 或 `plan` 要取代相同工作時可加入 `--force`。自訂參數請複製
`config.example.json`，再使用 `--config your-config.json`。摘要也可用
`--model MODEL` 暫時覆寫模型。

## 處理架構

```text
影片 → 靜音剪輯 → edited.mp4 → 本機 Whisper → transcript.json / .srt
                                                ├→ 固定規則排版 → transcript.md
                                                ├→ 原始逐字稿 → Qwen 英文摘要 → Qwen 繁中翻譯
                                                │                              ↓
                                                │                    雙語 Markdown / JSON
                                                └→ 字幕副本
                                                    ├→ 全文候選詞挖掘 → Qwen
                                                    │                  ├→ 全文規則
                                                    │                  └→ 區段規則
                                                    └→ 原始 ASR → Forced Aligner
                                                                       │
                                                     校正文字投影到原始時間軸
                                                                       ↓
                                                      稽核 → ASS / SRT / subtitled.mp4
```

ASR 維持原本設定，沒有加入 glossary、hotwords 或硬編碼的專名替換規則。
`transcript.json` 永遠保存原始 Whisper 結果，摘要也只讀這份原始內容。
`--subtitles` 才會建立獨立副本。程式先從整份逐字稿挖掘技術詞候選項，再讓本機
Qwen 在一次呼叫中提出兩類規則：重複錯誤使用全文規則，單一區段才成立的修正使用
區段規則。候選詞挖掘只負責縮小檢查範圍，不會自行決定替換內容；所有規則還須通過
精確來源匹配、近似發音與字數差等保守驗證，不會改寫句意。

Forced Aligner 永遠接收未修改的原始 ASR 文字，校正後的顯示文字是在對時完成後才
投影到時間軸，因此拼字替換不會造成對時失敗。字幕是 best effort：校正失敗時使用
原始 ASR 文字；Forced Aligner 整體或某段失敗時，退回 Whisper 的 word timestamps，
但已驗證的顯示文字仍會保留。最後會確認每個選定的修正都恰好送入一個字幕 cue；
字幕分支失敗也不會使已完成的剪輯、逐字稿或摘要失效。燒錄另存為
`subtitled.mp4`，音訊直接複製，不會覆蓋 `edited.mp4`。

字幕相關的自訂設定如下；`correction_max_rules` 已由用途更明確的兩個設定取代：

```json
{
  "subtitles": {
    "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
    "correction_context_tokens": 65536,
    "correction_output_tokens": 2048,
    "correction_candidate_limit": 48,
    "correction_rule_safety_cap": 32,
    "alignment_chunk_seconds": 120
  }
}
```

`correction_candidate_limit` 是送給 Qwen 檢查的全文候選詞上限，不等於實際修正數；
`correction_rule_safety_cap` 才是全文規則與區段規則合計可接受的安全上限。

## 摘要格式

摘要固定為 detailed overview，不再區分 standard/detailed，也不再產生
Key Takeaways、Uncertainties、Sections 或 Action Items。

兩個階段是兩次獨立的本機 Ollama 呼叫。第一階段讀取完整逐字稿視窗，只輸出：

```json
{
  "title_en": "...",
  "overview_en": ["..."],
  "completion_marker": "complete"
}
```

第二階段不再讀取 ASR 逐字稿，只以第一階段產生的英文 JSON 為來源，輸出：

```json
{
  "title_zh_tw": "...",
  "overview_zh_tw": ["..."],
  "completion_marker": "complete"
}
```

程式最後只按欄位組合兩份內容，不會改寫任何句子。英文與繁中段落數必須完全
一致。兩次呼叫都使用 non-thinking，讓有限的輸出 token 用在可見內容；兩階段
固定使用相同 `num_ctx`，讓 Ollama 可以重用已載入的 runner，翻譯完成就卸載。
`summary.metrics.json` 會分別記錄兩階段的 `load_duration_ns`，可核對第二階段是否
真的免除權重重載。

Prompt 會依逐字稿資訊量自動設定篇幅。以目前兩支影片的逐字稿計算，約 6,457
個來源英文字時，英文目標為 1,300–1,800 words、13–18 段；約 9,424 個來源
英文字時，目標為 1,900–2,600 words、16–22 段。每個有實質內容的時間視窗都
必須有所貢獻，實作型課程會優先保留指令、設定、除錯步驟與示範流程。

篇幅目標只會記錄在 metrics，不會成為容易卡住流程的硬性閘門。發布前只檢查：

- JSON 與完成標記完整。
- 至少 8 組英中對齊段落。
- 英文至少 1,000 words。
- 英文與繁中欄位使用正確文字系統。

程式不會修補、改寫或人工加工模型文字。`summary.en.raw.txt` 與
`summary.zh-TW.raw.txt` 分別保存兩次本機模型的原始回應；
`summary.metrics.json` 保存兩者的 SHA-256、第二階段英文來源的 SHA-256，以及
`post_generation_content_modified=false`。重新摘要時，所有新檔先寫入暫存目錄，
兩階段都驗證成功才取代上一版，失敗不會混出「新英文搭舊中文」的結果。

## 逐字稿格式

- `transcript.json` 是 canonical ASR 結果，保留所有 segment 與時間戳。
- `transcript.srt` 保留原始 ASR segmentation，適合字幕工具。
- `transcript.md` 每五分鐘分節，將零碎 segment 依停頓、長度與句尾合併成段落。
- Markdown 排版只正規化空白與標點前空格，不刪字、不換字、不去重，也不用 LLM。
- `transcript` 指令只讀既有 `transcript.json`，因此不會使用 Whisper、Qwen 或 GPU。

## 預設本機模型

- ASR：`faster-whisper-large-v3-turbo`，CUDA FP16。
- 摘要與翻譯：`qwen3.6:27b`，Ollama non-thinking，依序執行兩次。
- 字幕校正：相同的本機 `qwen3.6:27b`，一次輸出全文與區段 correction rules。
- 字幕對時：`Qwen3-ForcedAligner-0.6B`，CUDA BF16、120 秒切塊。
- 每階段輸出上限由內容目標計算；目前目標範圍會預留 8,192 tokens，設定的硬上限
  是 16,384 tokens。
- 專案專用 Ollama：`0.32.0`，`http://127.0.0.1:11435`。
- 程式要求 Ollama 0.32.0 以上、強制 loopback，並拒絕 `:cloud`／`-cloud` 模型。
- user service 設定 `OLLAMA_NO_CLOUD=1`；即使使用 localhost API，也不能把 prompt
  轉送到 Ollama Cloud。

這台機器上的服務名稱是 `orin-video-editor-ollama.service`：

```bash
systemctl --user status orin-video-editor-ollama.service
systemctl --user restart orin-video-editor-ollama.service
```

## 輸出

完整流程會建立 `output/<影片名稱>-<來源指紋>-<設定指紋>/`：

- `edited.mp4`
- `edit_plan.json`
- `transcript.json`、`transcript.srt`、`transcript.md`
- `summary.en.md`、`summary.zh-TW.md`
- `summary.json`、`summary.en.raw.txt`、`summary.zh-TW.raw.txt`
- `summary.metrics.json`
- `manifest.json` 與 FFmpeg log

加入 `--subtitles` 時另外產生：

- `subtitled.mp4`（燒錄字幕、保留音訊軌）
- `subtitle.srt`、`subtitle.ass`
- `subtitle.rules.json`、`subtitle.corrected.json`
- `subtitle.correction.raw.txt` 與對時／燒錄紀錄

`subtitle.rules.json` 保存全文候選詞、通過驗證的兩類規則、各規則的投影／cue
交付數量與整體 delivery audit，也明確記錄 canonical transcript、摘要輸入及
Forced Aligner 輸入均未被修改。

`--edit-only` 使用獨立工作 ID，只輸出剪輯相關檔案，不會和完整流程的快取混淆。

## 專案結構

```text
cli.py         CLI 入口
pipeline.py    串接剪輯、ASR 與摘要
media.py       FFmpeg probe、剪輯與音訊抽取
silence.py     靜音區間與剪輯計畫
asr.py         faster-whisper
transcript.py  逐字稿輸出、好讀排版與摘要時間視窗
summary.py     Ollama 兩階段摘要、結構檢查與 Markdown
subtitles.py   全文／區段校正規則、Forced Aligner、投影與字幕稽核
config.py      設定驗證
io_utils.py    原子寫檔與工作指紋
```

建置與測試：

```bash
./scripts/build_image.sh
./run_docker.sh test
```
