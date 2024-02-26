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

# Apriltag Detector
detector = Detector()

# Pose Detection Variables
nRows = 2
nCols = 2
termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]])
cubeCorners = np.float32([[0,0,0],[0,3,0],[3,3,0],[3,0,0],
                          [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    

def detect_apriltags(gray, frame):
    detections = detector.detect(gray)

    for detection in detections:
        corners = detection.corners.astype(int)
        id_num = detection.tag_id

        # Draw tag outline
        for i in range(len(corners)):
            cv2.line(frame, tuple(corners[i - 1]), tuple(corners[i]), (0, 255, 0), 2)

        # Display tag ID
        cv2.putText(frame, str(id_num), (corners[0][0], corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate pose
        tag_corners = np.array(corners, dtype=np.float32)

        _, rvecs, tvecs = cv2.solvePnP(worldPtsCur, tag_corners, camera_matrix, None)
        
        imgpts,_ = cv2.projectPoints(cubeCorners, rvecs,tvecs, camera_matrix, dist_cooef)

        imgpts = np.int32(imgpts).reshape(-1,2)

        print(f"imgpts is {imgpts}")

        #print(imgpts)

        # Add green plane 
        #img = cv2.drawContours(frame,[imgpts[:4]],-1,(0,255,0),-3)

        # Add box borders  
        #for i in range(4):
        #    j = i + 4
        #    img = cv2.line(frame,tuple(imgpts[i]),tuple(imgpts[j]),(255),3)
        #    img = cv2.drawContours(frame,[imgpts[4:]],-1,(0,0,255),3)

#==============================================
# Main Process
#==============================================        
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera image detected")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detect_apriltags(gray, frame)

        cv2.imshow("Apriltag Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

