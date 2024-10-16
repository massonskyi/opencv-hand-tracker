import numpy as np
import cv2
import mediapipe as mp
import time
import math
from pulsectl import Pulse

# Настройка MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Настройка камеры
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Настройка аудио
pulse = Pulse('volume-control')
sinks = pulse.sink_list()
default_sink = sinks[0]  # Assuming the first sink is the default one
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

# Инициализация MediaPipe Hands
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Обработка изображения
        imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        results = hands.process(imgRGB)
        imgRGB.flags.writeable = True
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = img.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Получение координат
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * image_width), int(lm.y * image_height)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    # Найти расстояние между указательным пальцем и большим пальцем
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    length = math.hypot(x2 - x1, y2 - y1)

                    # Преобразование громкости
                    volBar = np.interp(length, [50, 200], [400, 150])
                    volPer = np.interp(length, [50, 200], [0, 100])
                    smoothness = 10
                    volPer = smoothness * round(volPer / smoothness)

                    # Установка громкости
                    try:
                        pulse.volume_set_all_chans(default_sink, volPer / 100)
                        print(f"Volume set to: {volPer}%")
                    except Exception as e:
                        print(f"Error setting volume: {e}")

                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Отображение громкости
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cVol = int(pulse.volume_get_all_chans(default_sink) * 100)
        cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, colorVol, 3)

        # Отображение FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        # Показ изображения
        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
