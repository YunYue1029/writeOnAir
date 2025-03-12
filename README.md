# writeOnAir

概述

本專案利用 OpenCV 和 MediaPipe 實現基於手勢的繪圖應用。使用者可以通過攝像頭偵測手勢來 畫圖、擦除 以及 切換顏色。

功能

手部偵測：使用 MediaPipe 偵測單隻手。

繪圖模式：當食指和中指併攏時，可進行繪圖。

擦除模式：當所有手指彎曲時，啟動橡皮擦功能。

顏色切換：根據手指伸直的不同組合來變更畫筆顏色。

## 安裝方式
確保已安裝 Python，然後執行以下指令安裝所需的依賴庫：
```
pip install opencv-python mediapipe numpy
```
## 使用方法
運行程式：
```
python hand_gesture_drawing.py
```
## 退出程式
按下 " q " 鍵即可
