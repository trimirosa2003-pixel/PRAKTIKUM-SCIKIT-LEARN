import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Fungsi hitung sudut
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert warna
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Ambil titik kanan
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)

            # Klasifikasi
            if angle > 160:
                status = "BERDIRI"
                color = (0, 255, 0)
            else:
                status = "DUDUK"
                color = (0, 0, 255)

            cv2.putText(image, status,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color, 3)

            cv2.putText(image, f'Angle: {int(angle)}',
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,255,255), 2)

        except:
            pass

        # Gambar skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Deteksi Berdiri / Duduk', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()