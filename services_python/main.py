import cv2
import mediapipe as mp
import random
import time
from enum import Enum
import numpy as np
import json

class ShapeType(Enum):
    YOGA = "yoga"
    SPORTS = "sports"
    GEOMETRIC = "geometric"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

current_shape = None
scores = []
shape_change_time = time.time()
shape_display_duration = 3
similarity_threshold = 50

def select_shape_type():
    print("Select the type of shape:")
    print("1: Yoga")
    print("2: Sports")
    print("3: Geometric")

    choice = input("Enter your choice (1-3): ")
    shape_type_map = {
        "1": ShapeType.YOGA,
        "2": ShapeType.SPORTS,
        "3": ShapeType.GEOMETRIC
    }

    return shape_type_map.get(choice, ShapeType.YOGA)

def calculate_score(detected_coords, target_coords, target_shape_name):
    distances = []

    for detected, target in zip(detected_coords, target_coords):
        dist = np.linalg.norm(np.array(detected) - np.array(target))
        distances.append(dist)

    avg_distance = np.mean(distances)

    if target_shape_name == "Jumping":
        return 100

    if avg_distance < 20:
        return 100
    elif avg_distance < similarity_threshold:
        return max(0, int(100 - (avg_distance / (similarity_threshold * 0.5)) * 100))
    else:
        return random.randint(20, 60)

with open('resources/shapes.json', 'r') as f:
    shapes = json.load(f)

current_type = select_shape_type()
current_shape = random.choice(shapes[current_type.value])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    detected_coords = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

        for landmark in results.pose_landmarks.landmark:
            detected_coords.append((landmark.x * frame.shape[1], landmark.y * frame.shape[0]))

        if time.time() - shape_change_time > shape_display_duration:
            score = calculate_score(detected_coords, current_shape["coords"], current_shape["name"])
            scores.append((current_shape["name"], score))
            print(f"Score obtained for '{current_shape['name']}': {score}")

            current_shape = random.choice(shapes[current_type.value])
            shape_change_time = time.time()

        for coord in current_shape["coords"]:
            cv2.circle(frame, coord, 5, (255, 255, 0), -1)
        cv2.putText(frame, current_shape["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Mimic the shape!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow('Real-time Body Pose Detection', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Scores obtained in the session:")
for shape_name, score in scores:
    print(f"{shape_name}: {score}")
