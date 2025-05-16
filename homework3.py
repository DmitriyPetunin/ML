import cv2
import mediapipe as mp
import os
from deepface import DeepFace

# Инициализация MediaPipe для рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Распознаем только одну руку
mp_drawing = mp.solutions.drawing_utils

# Инициализация OpenCV для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка вашего изображения (замените 'your_photo.jpg' на путь к вашей фотографии)
your_image_path = "/Users/dmitriypetunin/Documents/photo.jpeg"
print(os.path.exists(your_image_path))

# Функция для подсчета поднятых пальцев
def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]  # Индексы кончиков пальцев
    count = 0

    # Проверка большого пальца
    if landmarks[4].x < landmarks[3].x:
        count += 1

    # Проверка остальных пальцев
    for tip in finger_tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    return count

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертация кадра в RGB для MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка рук
    hand_results = hands.process(rgb_frame)
    fingers_count = 0


    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks.landmark)

    # Отображение количества пальцев в углу экрана
    cv2.putText(
        frame,
        f"Fingers: {fingers_count}",
        (10, 30),  # Позиция текста (верхний левый угол)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Размер шрифта
        (0, 0, 255),  # Цвет текста (красный)
        2,  # Толщина текста
    )

    # Обработка лиц
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Определение эмоции лица
        try:
            emotion = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion[0]['dominant_emotion']
        except Exception:
            dominant_emotion = "Неизвестная эмоция"

            
        try:
            faces = DeepFace.extract_faces(your_image_path)
            # print(faces)  # Should return a list of detected faces
        except Exception as e:
            print("Face not detected:", e)

        # Проверка, ваше ли это лицо
        try:
            result = DeepFace.verify(
                img1_path=your_image_path,
                img2_path=frame[y:y+h, x:x+w],
                model_name="Facenet",  # Модель для сравнения лиц
                distance_metric="cosine"  # Метрика расстояния
            )
            # print("Verification result:", result)
            is_owner = result["verified"]
        except Exception as e:
            # print("Error during verification:", e)
            is_owner = False

        name = "unknown"

        print("is_owners = " ,is_owner)

        if is_owner:

            print("fingers_count =", fingers_count)

            if fingers_count == 1:
                print("зашло в условие 1")
                name = "Dmitriy" 
            elif fingers_count == 2:
                print("зашло в условие 2")
                name = "Petunin" 
            elif fingers_count == 3:
                print("зашло в условие 3")
                name = dominant_emotion 
            else:
                name = ""  # Ничего не выводим
        else:
            name = "unknown"  # Лицо не принадлежит владельцу

        if name:
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        # Подпись имени или фамилии
        # if fingers_count == 1 or fingers_count == 2:
        #     cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # # Подпись эмоции, если три пальца
        # if fingers_count == 3:
        #     cv2.putText(frame, dominant_emotion, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow("Face and Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()

cv2.destroyAllWindows()