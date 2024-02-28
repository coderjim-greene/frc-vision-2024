"""
File: detect_notes_mt.py
Author: Jim Greene
Date: 3-Feb-2023

Description: Multi-threaded FRC 2024 Note detection application.  
             Uses OpenCV to capture video frames, and detects the orange Note torus objects 
             used in the FRC 2024 game, and stores them in an array of dictionary objects 
             called detected_notes.  Each item in this array contains the width, height, area, 
             estimated distance from the camera (using camera focal length as the basis), and the
             angle the detected notes is oriented from the center of the image.

             Capturing of frames, and image recognition processing occur in separate threads.  Images 
             frames are captured in a standalone thread, and communicated back to the main thread using 
             a queue.  This was done as an experiment to see if the image processing performance would be 
             improved by separating obtaining images from the camera (which is a blocking call), and the actual 
             OpenCV processing of the image frames.
"""

import cv2
import numpy as np
import threading
from collections import deque
from typing import List


# Size of actual note object, in inches
note_size_inches = 14
note_size_meters = note_size_inches * 0.0254
focal_length = 930

class VideoCaptureThread(threading.Thread):
    cap: cv2.VideoCapture
    video_queue: deque
    stop_thread_event: threading.Event
    frame_width: int
    frame_height: int
    fps: float

    def __init__(self, video_source: int, video_queue: deque, stop_thread_event: threading.Event, name="Video Capture Thread"):
        threading.Thread.__init__(self)

        self.video_queue = video_queue
        self.stop_thread_event = stop_thread_event
        self.cap = cv2.VideoCapture(video_source)

        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # Set height

        # Set the FPS for the camera
        self.cap.set(cv2.CAP_PROP_FPS, 20)  

        # Read the first fame
        ret, frame = self.cap.read()

        # Store the attributes of the frame (dimensions, fps)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fps = self.cap.get(5)

    def run(self):
        print("Starting video capture thread...")
        while True:
            ret, frame = self.cap.read()
            self.video_queue.appendleft(frame)

            if self.stop_thread_event.is_set():
                print("Detected stop_thread_event...")
                self.cap.release()
                cv2.destroyAllWindows()
                return
        


    def get_frame_details(self):
        return {
            "frame_width": self.frame_width, 
            "frame_height": self.frame_height,
            "fps": self.fps, 
            "crosshair_x": int(self.frame_width / 2),
            "crosshair_y": int(self.frame_height / 2)
        }

def estimate_distance(object_width_pixels):
    """
    Estimates distance of detected objects based on the width of the detected object (in pixels), 
    the known size of the notes (14in), and the focal length of the webcam.
    Returns:  distance (in meters)
    """
    # Estimate distance using the formula: distance = (focal_length * real_object_width) / object_width_pixels
    # Returns: meters
    distance = (focal_length * note_size_meters) / object_width_pixels
    return distance

def calculate_angle(frame, object_x):
    """
    Calculates the angle the detected object is from the center of the image.
    Returns:  tuple (direction in ["R", "L"], degrees to turn)
    """
    # Get the center of the image frame
    height, width, _ = frame.shape
    center_x = width // 2

    # Calculate horizontal distance between object and center
    distance_from_center = center_x - object_x

    # Calculate angle using trigonometry
    # Assuming the camera is directly facing the object
    angle_radians = np.arctan(distance_from_center / focal_length)
    angle_degrees = np.degrees(angle_radians)

    # If the object_x value is less than the center of the image, set the 
    # direction to "L" [left], otherwise the direction is right.
    direction = "R"
    if object_x < center_x:
        direction = "L"

    # By default, the angle is reversed from the direction of the image.
    # Multiply this value by -1, so all items left of center have a negative angle value , 
    # and all items right of the center have a postive angle value.
    angle_degrees = round(angle_degrees * -1, 4)

    return direction, angle_degrees

def detect_notes(frame, frame_details: dict) -> List[dict]:
    # Calculate the coordinates to be used to draw a global crosshair over each image frame, 
    # to give the operator an idea of where the camera is pointing.
    detected_notes = []

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color
    lower_orange = np.array([0, 102, 204])
    upper_orange = np.array([20, 255, 255])

    # Create a mask using the inRange function
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a morphological operation to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add details to frame
    cv2.putText(frame, f'FPS:  {frame_details["fps"]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (36,255,12), 2)

    # Draw frame crosshair
    cv2.line(frame, (frame_details["crosshair_x"], 0), (frame_details["crosshair_x"], frame_details["frame_height"]), (0, 0, 255), 1)
    cv2.line(frame, (0, frame_details["crosshair_y"]), (frame_details["frame_width"], frame_details["crosshair_y"]), (0, 0, 255), 1)

    # Draw bounding boxes around detected contours
    note_id = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w > 2*h) and (w*h > 2000):
            note_center_x = x + int(w/2)
            note_center_y = y + int(h/2)

            # Calculate range and angle for detected object
            range = estimate_distance(w)
            angle_data = calculate_angle(frame, note_center_x)

            note = {"id": note_id, "width": w, "area": w*h, "range": range}
            detected_notes.append(note)
            note_id += 1

            # Draw bounding box and crosshair in center of detected object
            if angle_data[1] > -0.2 and angle_data[1] < 0.2:
                # Blue Box - Blue crosshair
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.line(frame, (note_center_x - 10, note_center_y), (note_center_x + 10, note_center_y), (255, 0, 0), 2)
                cv2.line(frame, (note_center_x, note_center_y - 10), (note_center_x, note_center_y + 10), (255, 0, 0), 2)
            else:
                # Green box - Red crosshair
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.line(frame, (note_center_x - 10, note_center_y), (note_center_x + 10, note_center_y), (0, 0, 255), 2)
                cv2.line(frame, (note_center_x, note_center_y - 10), (note_center_x, note_center_y + 10), (0, 0, 255), 2)

            # Add metrics text to detected object
            # Text on top of detected object
            cv2.putText(frame, f'ID[{note["id"]}], W[{w}], H[{h}], A[{note["area"]}]', (x+5, y-22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            # Text on bottom of detected object
            cv2.putText(frame, f'Range[{round(range * 39.37, 2)}in], Angle:[{angle_data[0]}/{angle_data[1]}deg]', (x+5, (y+h)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

    # Sort the array of found notes by the area, so the largest items are at the top
    detected_notes = sorted(detected_notes, key=lambda x: x['area'])

    # Display the frame with bounding boxes
    cv2.imshow("FRC 2024 Note Detection", frame)

    return detected_notes


# Start Processing
if __name__ == "__main__":
    # Queue to store frames from the running camera_thread
    frame_queue = deque(maxlen=1)
    # Thread-safe boolean to enable telling the thread to stop processing
    stop_thread_event = threading.Event()

    # Python thread that initiates the video capture, and stores image frames in frame_queue
    camera_thread = VideoCaptureThread(video_source=0, video_queue=frame_queue, stop_thread_event=stop_thread_event, name="video_capture_thread")
    camera_thread.start()

    # Capture the details about the frames (dimensons, frames per second from the camera)
    frame_details = camera_thread.get_frame_details()

    while True:
        if len(frame_queue) > 0:
            frame = frame_queue.pop()
            if frame is not None:
                # perform image recognition 
                detect_notes(frame, frame_details)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_thread_event.set()
            break
