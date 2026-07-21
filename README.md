# Whisper AI 語音轉文字工具

本專案是一個本機 Streamlit 介面，用 OpenAI Whisper 將影片或音檔轉成字幕與文字檔。介面會先顯示音訊波形，可只轉錄指定 A–B 範圍，輸出的字幕時間仍對應原始影片。

## 主要功能

- 影片或音訊載入後顯示簡化音訊波形，可快速看出長時間靜音。
- 在波形上水平拖曳選取 A–B，橫軸與精確輸入都使用 `HH:MM:SS`。
- 拖曳只更新暫存範圍；按下「確認選取」才正式套用，按「轉錄選取範圍」才擷取音訊並執行 Whisper。
- 只擷取選定範圍交給 Whisper，避免被前後長靜音干擾。
- SRT、VTT、TSV、JSON 的時間會自動加回 A 點，因此可直接套回原影片。
- 預設使用 OpenCC `s2twp`，將簡體轉為台灣繁體與常用詞後再輸出。
- 支援 TXT、SRT、VTT、TSV、JSON。

## 安裝

建議在 Conda 環境中安裝套件：

```powershell
pip install -r requirements.txt
```

Whisper 另外需要系統已安裝 FFmpeg，並且 `ffmpeg` 可以在終端機中直接執行。

## 啟動

雙擊 `run_whisper.bat`。這個檔案會呼叫 `start_whisper.ps1` 來啟動 Conda 與 Streamlit。

也可以手動執行：

```powershell
streamlit run whisper_app.py --server.maxUploadSize 5000
```

## 使用建議

- 大型影片建議使用「直接輸入電腦中的檔案路徑」，可避免瀏覽器上傳造成記憶體壓力。
- 在波形上由左往右拖曳即可暫存 A–B；綠線代表開始、紅線代表結束。確認位置後再按「確認選取」。
- 要精確指定位置時，直接在下方輸入開始與結束時間，例如 `00:18:30`。
- 選段輸出的檔名會附上 A–B 範圍，但字幕內時間碼仍是原影片時間。
- 如果不需要簡轉繁，可在側欄的「中文字體輸出」選擇「保留 Whisper 原文」。
- 上傳模式可自行勾選是否在轉換完成後清除原始暫存檔；預設保留，方便再次調整範圍。
- Whisper 模型會下載到專案內的 `models/` 資料夾。
- 模型會在同一次 Streamlit 工作階段中快取，連續轉檔時不用每次重新載入。
