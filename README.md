# Whisper AI 語音轉文字工具

本專案是一個本機 Streamlit 介面，將影片或音檔轉成字幕與文字檔。介面採用「選擇媒體 → 選擇片段 → 轉譯與輸出」三步驟流程，可只轉錄指定 A–B 範圍，輸出的字幕時間仍對應原始影片。支援兩種辨識引擎：OpenAI Whisper 與 SenseVoice-small（FunASR）。

## 主要功能

- 影片或音訊載入後顯示簡化音訊波形，可快速看出長時間靜音。
- 在波形上水平拖曳即可套用 A–B，並支援 `HH:MM:SS.mmm` 毫秒時間碼。
- 可試聽選取起點 30 秒，確認內容後再執行辨識。
- 只擷取選定範圍交給辨識引擎，避免被前後長靜音干擾。
- SRT、VTT、TSV、JSON 的時間會自動加回 A 點，因此可直接套回原影片。
- 預設使用 OpenCC `s2twp`，將簡體轉為台灣繁體與常用詞後再輸出。
- 支援 TXT、SRT、VTT、TSV、JSON。
- 提供 Turbo、Medium、Large v3 等速度／品質預設。
- 轉譯結果會保留在工作階段，可編輯字幕或新增輸出格式而不必重新推論。
- 顯示音訊擷取、模型載入、推論、後處理及 realtime 倍率。

## 辨識引擎

側欄「辨識引擎」可切換：

- **Whisper（OpenAI）**：可選 Turbo／Medium／Large v3 等速度與品質預設，語言用下拉選擇（中文、自動偵測、英文、日文、韓文），並可填入「專有名詞提示」。模型下載到專案內的 `models/` 資料夾。
- **SenseVoice-small（中英混合推薦）**：阿里 FunASR 的開源 ASR 模型，對中英夾雜語音的辨識效果比 Whisper 好、推論也快。語言可選「自動偵測」（中英混合建議）或指定中文／英文／粵語／日文／韓文。首次使用會從 Hugging Face 下載約 1GB 的模型與 VAD 模型（`FunAudioLLM/SenseVoiceSmall` + `funasr/fsmn-vad`），之後在本機（`%USERPROFILE%\.cache\huggingface`）快取。字幕分段依標點與最長時長／字數自動切段，時間碼取自模型的逐字時間戳記。

## 安裝

建議建立 Python 3.11 Conda 環境後安裝套件：

```powershell
conda create -n turnsrt python=3.11 -y
conda activate turnsrt
pip install -r requirements.txt
```

Whisper 另外需要系統已安裝 FFmpeg，並且 `ffmpeg` 可以在終端機中直接執行。

## 啟動

雙擊 `run_whisper.bat`。啟動器會優先從 PATH 自動尋找 Conda，再呼叫 `start_whisper.ps1` 檢查 Python 套件與 FFmpeg，最後啟動 Streamlit。若要固定使用上面的獨立環境，可先設定使用者環境變數 `WHISPER_CONDA_ENV=turnsrt`；未設定時會使用 Conda base 環境。

也可以手動執行：

```powershell
streamlit run whisper_app.py --server.maxUploadSize 5000
```

## 使用建議

- 大型影片建議使用「直接輸入電腦中的檔案路徑」，可避免瀏覽器上傳造成記憶體壓力。
- 在波形上由左往右拖曳即可套用 A–B；綠線代表開始、紅線代表結束。
- 要精確指定位置時，直接在下方輸入開始與結束時間，例如 `00:18:30.500`。
- 選段輸出的檔名會附上 A–B 範圍，但字幕內時間碼仍是原影片時間。
- 每個來源會輸出到 `outputs/<檔名_來源識別碼>/`，避免同名媒體互相覆蓋。
- 如果不需要簡轉繁，可在側欄的「中文字體輸出」選擇「保留 Whisper 原文」。
- 人名、品牌或術語可填入「專有名詞提示」，提高固定詞彙的一致性（僅 Whisper 引擎適用）。
- 中英夾雜的語音建議先試 SenseVoice-small（語言選「自動偵測」）；若需要更強語意理解或效果不理想，再切回 Whisper Large v3。
- 上傳模式可自行勾選是否在轉換完成後清除原始暫存檔；預設保留，方便再次調整範圍。
- Whisper 模型會下載到專案內的 `models/` 資料夾；SenseVoice 模型下載到使用者的 Hugging Face 快取。
- 同一時間只快取一個 Whisper 模型；切換模型時會淘汰舊項目，也可從側欄手動釋放 GPU 模型（會一併清除 SenseVoice）。
