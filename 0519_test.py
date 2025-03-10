import cv2
import mediapipe as mp
import numpy as np
import math

# 初始化手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 变量来跟踪绘图和橡皮擦状态
drawing = False
erasing = False
last_x, last_y = None, None

# 创建一个空白的画布用于记录绘图
canvas = None

# 定义判断角度阈值
angle_threshold = 10  # 根据需要调整此值
straight_threshold = 50  # 手指伸直的角度阈值
bent_threshold = 50  # 手指弯曲的角度阈值
eraser_radius = 50  # 橡皮擦半径

# 默认画笔颜色
current_color = (0, 255, 0)  # 绿色

# 计算两个2D向量之间的角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 180
    return angle_

# 计算手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 初始化画布大小
    if canvas is None:
        canvas = np.zeros_like(frame)

    # 将图像水平翻转
    frame = cv2.flip(frame, 1)

    # 将图像转换为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape

            # 获取手部坐标
            hand_points = []
            for id, lm in enumerate(hand_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                hand_points.append((px, py))

            # 计算手指角度
            angles = hand_angle(hand_points)

            # 获取手腕的坐标
            wrist_x, wrist_y = hand_points[0]

            # 检查手指是否全部弯曲
            if all(angle > bent_threshold for angle in angles):
                erasing = True
            else:
                erasing = False

            # 获取食指和中指指尖与指根的坐标
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            index_tip_x, index_tip_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            index_mcp_x, index_mcp_y = int(index_finger_mcp.x * w), int(index_finger_mcp.y * h)
            middle_tip_x, middle_tip_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            middle_mcp_x, middle_mcp_y = int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h)

            # 计算手指是否伸直
            index_straight = angles[1] < straight_threshold
            middle_straight = angles[2] < straight_threshold
            ring_straight = angles[3] < straight_threshold
            pinky_straight = angles[4] < straight_threshold

            # 检查并设置画笔颜色
            if middle_straight and ring_straight and pinky_straight:
                current_color = (255, 0, 0)  # 蓝色
            elif ring_straight and pinky_straight:
                current_color = (0, 255, 0)  # 绿色
            elif pinky_straight:
                current_color = (0, 0, 255)  # 红色
            

            # 检查食指和中指是否伸直且合拢
            index_vector = [index_tip_x - index_mcp_x, index_tip_y - index_mcp_y]
            middle_vector = [middle_tip_x - middle_mcp_x, middle_tip_y - middle_mcp_y]
            angle = vector_2d_angle(index_vector, middle_vector)

            if index_straight and middle_straight and angle < angle_threshold:
                drawing = True
            else:
                drawing = False

            # 绘图或橡皮擦功能
            if drawing and not erasing:
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (index_tip_x, index_tip_y), current_color, 5)
                last_x, last_y = index_tip_x, index_tip_y
            elif erasing:
                cv2.circle(frame, (wrist_x, wrist_y), eraser_radius, (0, 0, 0), -1)
                cv2.circle(canvas, (wrist_x, wrist_y), eraser_radius, (0, 0, 0), -1)
                last_x, last_y = None, None

            # 绘制指尖，颜色跟随画笔颜色
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, current_color, cv2.FILLED)
            cv2.circle(frame, (middle_tip_x, middle_tip_y), 10, current_color, cv2.FILLED)

            # 显示手部标记
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 将画布叠加到当前的帧
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # 显示图像
    cv2.imshow('Hand Gesture Drawing', frame)

    # 检测按键 'q' 退出
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
