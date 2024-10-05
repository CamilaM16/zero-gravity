import cv2
import mediapipe as mp

class PoseHandler:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
        self.drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
