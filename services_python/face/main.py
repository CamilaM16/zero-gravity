import cv2
from deepface import DeepFace
import mediapipe as mp
import time
import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
life_time = 10

emotions_to_detect = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
target_size = (screen_width // 4, screen_height // 4)

start_time = time.time()
current_emotion = random.choice(emotions_to_detect)

def get_random_position():
    x = random.randint(0, screen_width - target_size[0])
    y = random.randint(0, screen_height - target_size[1])
    return (x, y)

def is_face_in_target(face_coords, target_coords):
    face_x, face_y, face_w, face_h = face_coords
    target_x, target_y = target_coords

    face_right = face_x + face_w
    face_bottom = face_y + face_h
    target_right = target_x + target_size[0]
    target_bottom = target_y + target_size[1]

    face_area = face_w * face_h
    target_area = target_size[0] * target_size[1]

    overlap_x = max(0, min(face_right, target_right) - max(face_x, target_x))
    overlap_y = max(0, min(face_bottom, target_bottom) - max(face_y, target_y))
    
    overlap_area = overlap_x * overlap_y

    return overlap_area >= (0.5 * target_area)

target_position = get_random_position()

while life_time > 0:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    cv2.rectangle(frame, target_position, (target_position[0] + target_size[0], target_position[1] + target_size[1]), (255, 0, 0), 2)
    cv2.putText(frame, f"Do: {current_emotion.capitalize()}", (target_position[0], target_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    results = pose.process(rgb_frame)

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if is_face_in_target((x, y, w, h), target_position) and emotion == current_emotion:
                life_time += 3 
                current_emotion = random.choice(emotions_to_detect)
                target_position = get_random_position()
                start_time = time.time()
        except Exception as e:
            print(f"Error analizando la emoción: {str(e)}")

    elapsed_time = time.time() - start_time
    remaining_time = life_time - elapsed_time

    cv2.putText(frame, f"Time: {int(remaining_time)}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if remaining_time <= 0:
        print("Game Over")
        break

    cv2.imshow('Emotion Challenge', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
