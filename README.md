0429更新重點-新增一個main_test.py作為功能測試(9453cc沛佳)
功能修改與除錯
1.fix視窗無法關閉的問題
2.fix換新的teacher_dance.mp4但dance_standard.json不會跟著做修改,會導致老師的骨架呈現錯誤
3.add空白鍵在s啟動之後可以做暫停與撥放,c可以叫出或是收回倍速調整視窗(x0.5/x1.0/x1.5/x2.0)
-------------------------------------------------------------------------------------------
# senior_project
檔案用途說明
app.py
這是一個簡單的即時姿態辨識演示程式（PoC）。它使用MediaPipe的Pose模型從攝影機（WebCam）讀取影像，偵測人體骨架，並在畫面上繪製骨架線條和關節點。用途是測試基本的姿態辨識功能，顯示即時的骨架追蹤。沒有評分邏輯，只是純粹的視覺化。

config.py
設定檔案，定義了MediaPipe模型的路徑、信心度閾值（detection/tracking）、繪圖顏色和骨架連線定義。用途是集中管理所有常數和參數，便於調整系統行為（如信心度或視覺樣式）。

dance_standard.json
這是一個JSON檔案，儲存從老師舞蹈影片中提取的標準骨架資料。每個幀（frame）包含時間戳記和33個關鍵點的座標（x, y, z, visibility）。用途是作為評分基準，讓系統比較學生的姿態與標準動作的差異。

extract_standard.py
工具腳本，用於從老師的舞蹈影片（teacher_dance.mp4）中提取標準骨架資料。它使用MediaPipe處理影片每一幀，儲存骨架座標到dance_standard.json。用途是預處理階段，建立評分用的標準資料庫。

extractor.py
一個類別（PoseExtractor），封裝了MediaPipe的PoseLandmarker模型，用於從影像中提取骨架資料。支援影片模式（VIDEO），並設定信心度。用途是模組化地處理姿態提取，供其他檔案調用。

main.py
主要應用程式，實現了舞蹈教學系統的核心邏輯。它同時處理WebCam（學生）和老師影片，進行即時評分。按下's'開始播放影片並評分，按下'q'退出。畫面會拼接顯示老師和學生的影像，並在學生畫面上顯示分數和時間。用途是整合所有功能，提供完整的互動體驗。

pose_utils.py
工具函數模組，包含骨架正規化（normalize_pose）和誤差計算（calculate_pose_error）。正規化會將骨架對齊骨盆中心並縮放，誤差計算比較學生與標準的距離。用途是處理骨架資料的數學運算，支援評分邏輯。

py_video.py
另一個視訊處理腳本，專注於膝蓋角度計算和蹲姿評分。它從兩個來源（舞蹈影片和WebCam）讀取影像，計算膝蓋角度，並根據角度範圍給予回饋（如"Good! Keep going"或"Squat down!"）。用途是針對特定動作（蹲姿）的即時指導，類似於健身應用。

README.md
專案說明檔案，通常包含專案概述、安裝步驟、使用說明和貢獻指南。用途是文檔化專案，讓新使用者快速上手。

visualizer.py
視覺化模組，定義了draw_skeleton函數，用於在畫布上繪製骨架（關節點和連線）。用途是將骨架資料轉換為視覺輸出，支援繪圖功能。

pycache/
Python的快取目錄，存放編譯後的.pyc檔案。用途是加速Python程式執行，無需手動管理。

整體資料流向
這個系統的資料流向大致如下（以main.py為核心）：

輸入階段：

WebCam（學生影像）和老師影片（teacher_dance.mp4）同時被讀取。
學生影像通過MediaPipe的Pose模型提取即時骨架資料（33個關鍵點的x, y, z, visibility）。
處理階段：

根據播放時間（elapsed_ms），從dance_standard.json中找到對應的標準骨架幀。
使用pose_utils.py將學生和標準骨架正規化（對齊骨盆、縮放）。
計算誤差（calculate_pose_error），轉換為分數（0-100）。
輸出階段：

在學生畫面上繪製骨架（使用visualizer.py或MediaPipe的繪圖工具）。
顯示分數、時間，並拼接老師和學生畫面。
如果是py_video.py，額外計算膝蓋角度並給予文字回饋。
總體流程：資料從影像→骨架提取→正規化比較→視覺化輸出。標準資料（dance_standard.json）是預先從老師影片提取的基準，而即時資料來自WebCam。系統支援即時評分和指導。
