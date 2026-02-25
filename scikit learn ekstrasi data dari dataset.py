import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=1)

DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "python", "buku")

data = []
labels = []

for dir_in in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_in)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_in)):
            if not img_path.lower().endswith(".jpg") :
                continue

        x = []
        y = []
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_in, img_path))

        if img is None:
            print("Gagal baca:", img_path)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x.append(hand_landmarks.landmark[i].x)
                    y.append(hand_landmarks.landmark[i].y)

                for i in range(len(x)):
                    data_aux.append(x[i] - min(x))
                    data_aux.append(y[i] - min(y))

            data.append(data_aux)
            labels.append(dir_in)

save_path = os.path.join(DATA_DIR, 'data.pickle')

with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Data berhasil disimpan!")