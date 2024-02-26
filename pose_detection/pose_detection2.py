import cv2
import numpy as np
from apriltag import Detector

def draw_axes(img, corners, imgpts):
    # Draw axes
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def main():
    cap = cv2.VideoCapture(0)

    detector = Detector()

    # Camera parameters (you need to calibrate your camera and use the intrinsic parameters here)
    camera_params = (600, 600, 320, 240)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        for detection in detections:
            # Center
            center = detection.center.astype(int)

            print(f"Center is {center}")

            # Draw bounding box around the tag
            corners = np.array(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
            frame = cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

            # Estimate pose and draw axes
            pose, e0, e1 = detector.detection_pose(detection, camera_params)
            imgpts, _ = cv2.projectPoints(np.array([(0,0,0),(3,0,0),(0,3,0),(0,0,3)], dtype=np.float32), pose[0], pose[1], np.eye(3), None)
            frame = draw_axes(frame, corners, imgpts.astype(int))

        cv2.imshow('Apriltag Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
