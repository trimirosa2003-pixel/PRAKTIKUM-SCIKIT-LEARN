import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model (BUKAN data.pickle!)
with open(r'C:\Users\user\Documents\Rosa\python\buku\pelatihan\model.pickle', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak bisa dibuka")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

labels_dict = {0: 'BUKA', 1: 'TUTUP'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            data_aux = np.array(data_aux).reshape(1, -1)

            prediction = model.predict(data_aux)
            predicted_character = labels_dict[int(prediction[0])]

            cv2.putText(
                frame,
                predicted_character,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

    cv2.imshow('Deteksi Tangan', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()