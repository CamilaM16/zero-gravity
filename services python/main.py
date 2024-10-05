import time

from shapes.shape_provider import ShapeProvider
from scoring.score_calculator import ScoreCalculator
from pose_detection.pose_handler import PoseHandler
from utils.camera import Camera

def main():
    shape_provider = ShapeProvider()
    score_calculator = ScoreCalculator()
    pose_handler = PoseHandler()
    camera = Camera()

    shape_provider.select_shape_type(input("Seleccione el tipo de forma (1-3): "))
    current_shape = shape_provider.get_random_shape()

    scores = []
    shape_change_time = time.time()
    shape_display_duration = 3

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        results = pose_handler.process_frame(frame)

        if results.pose_landmarks is not None:
            pose_handler.drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        else:
            print("No se detectaron puntos de pose.")

            if time.time() - shape_change_time > shape_display_duration:
                score = score_calculator.calculate_score()
                scores.append((current_shape["name"], score))
                print(f"Puntaje obtenido para '{current_shape['name']}': {score}")

                current_shape = shape_provider.get_random_shape()
                shape_change_time = time.time()

            for coord in current_shape["coords"]:
                cv2.circle(frame, coord, 5, (255, 255, 0), -1)
            cv2.putText(frame, current_shape["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Imita la forma!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow('Real-time Body Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Puntajes obtenidos en la sesión:")
    for shape_name, score in scores:
        print(f"{shape_name}: {score}")

if __name__ == "__main__":
    main()
