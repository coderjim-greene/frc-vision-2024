import cv2
import numpy as np
from apriltag import Detector

# Camera intrinsic parameters (to be calibrated)
camera_matrix = np.array([[1.51114732e+03, 0.00000000e+00, 8.13642618e+02],
                          [0.00000000e+00, 1.50861334e+03, 4.53750929e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], 
                          dtype=np.float32)

dist_cooef = np.array([[0.24441639,-0.47555724,-0.00478818,0.02840177,-0.86215734]], 
                      dtype=np.float32)

# AprilTag corner points (in tag frame)
tag_points_3d = np.array([[-0.5, -0.5, 0],
                           [0.5, -0.5, 0],
                           [0.5, 0.5, 0],
                           [-0.5, 0.5, 0]], dtype=np.float32)

# Initialize AprilTag detector
detector = Detector()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    detections = detector.detect(gray)

    for detection in detections:
        center = detection.center.astype(int)

        # Get the corners of the AprilTag
        corners = detection.corners.astype(int)
        id_num = detection.tag_id

        img_corners = np.array(corners, dtype=np.float32)

        # Solve PnP problem to estimate pose
        ret, rvec, tvec = cv2.solvePnP(tag_points_3d, img_corners, camera_matrix, None)

        # Draw axis on the tag
        axis_points, _ = cv2.projectPoints(np.array([[0, 0, 0],
                                                     [1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]], dtype=np.float32),
                                            rvec, tvec, camera_matrix, dist_cooef)
        
        axis_points = np.int32(axis_points).reshape(-1, 2)

        # Draw tag outline
        for i in range(len(corners)):
            cv2.line(frame, tuple(corners[i - 1]), tuple(corners[i]), (0, 255, 0), 3)

        # Display tag ID
        cv2.putText(frame, str(id_num), (corners[0][0], corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # calculate x and y diffences from the center of each tag to the z axis point
        x_diff = (axis_points[3][0] - center[0]) * -1
        y_diff = (axis_points[3][1] - center[1]) * -1

        # draw a cube over the tag, pointing in the direction the tag is facing
        for i in range(4):
            cv2.line(frame, corners[i], (corners[i][0] + x_diff, corners[i][1] + y_diff), (0, 0, 255), 3)
            if i > 0 & i < 3:
                cv2.line(frame, (corners[i][0] + x_diff, corners[i][1] + y_diff), (corners[i-1][0] + x_diff, corners[i-1][1] + y_diff), (0, 0, 255), 3)
            
            if i == 3:
                cv2.line(frame, (corners[i][0] + x_diff, corners[i][1] + y_diff), (corners[0][0] + x_diff, corners[0][1] + y_diff), (0, 0, 255), 3)

    # Show the updated image
    cv2.imshow('AprilTag Pose Estimation', frame)

    # Check to see if the user pressed Q to terminate the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
