"""
File: detect_notes_st.py
Author: Jim Greene
Date: 3-Feb-2023

Description: Single threaded FRC 2024 Note detection application.  
             Uses OpenCV to capture video frames, and detects the orange Note torus objects 
             used in the FRC 2024 game, and stores them in an array of dictionary objects 
             called detected_notes.  Each item in this array contains the width, height, area, 
             estimated distance from the camera (using camera focal length as the basis), and the
             angle the detected notes is oriented from the center of the image.
"""

import cv2
import numpy as np
from typing import List

# Size of actual note object, in inches
note_size_inches = 14
# The note size must be known in meters for the distance calculations
note_size_meters = note_size_inches * 0.0254
# Estimated focal length of the Microsoft Lifecam HD-3000 camera
focal_length = 930

def capture_frame_details(cap_object) -> dict:
    # Capture the size of the frame images, as well as the frames per second from the camera
    frame_width = int(cap_object.get(3))
    frame_height = int(cap_object.get(4))
    fps = cap_object.get(5)

    # Calculate the coordinates to be used to draw a global crosshair over each image frame, 
    # to give the operator an idea of where the camera is pointing.
    crosshair_x = int(frame_width / 2)
    crosshair_y = int(frame_height / 2)

    return {
        "frame_width": frame_width, 
        "frame_height": frame_height, 
        "fps": fps, 
        "crosshair_x": crosshair_x, 
        "crosshair_y": crosshair_y
    }


def estimate_distance(object_width_pixels):
    """
    Estimates distance of detected objects based on the width of the detected object (in pixels), 
    the known size of the notes (14in), and the focal length of the webcam.

    Returns:  distance: float [in meters]
    """
    # Estimate distance using the formula: distance = (focal_length * real_object_width) / object_width_pixels
    # Returns: meters
    distance = (focal_length * note_size_meters) / object_width_pixels
    return distance

def calculate_angle(frame, object_x):
    """
    Calculates the angle the detected object is from the center of the image.

    Returns:  tuple (direction: str in ["L", "R"], degrees: float [negative for left, positive for right])
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


def detect_orange_objects(video_source, frame, frame_details) -> List[dict]:
    # Reset the contents of this array with each iteration.
    detected_notes = []

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color
    lower_orange = np.array([0, 102, 204])
    upper_orange = np.array([20, 255, 255])

    # Create a mask using the inRange function
    # Effectively this is an image that is all black except for the pixels that match 
    # the lower_orange -> upper_orange ranges.
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a morphological operation to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add details to frame
    cv2.putText(frame, 
                f'FPS:  {frame_details["fps"]}', 
                (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.75, (36,255,12), 
                2)

    # Draw frame crosshair
    cv2.line(frame, 
             (frame_details["crosshair_x"], 0), 
             (frame_details["crosshair_x"], frame_details["frame_height"]), 
             (0, 0, 255), 
             1)
    
    cv2.line(frame, 
             (0, frame_details["crosshair_y"]), 
             (frame_details["frame_width"], frame_details["crosshair_y"]), 
             (0, 0, 255), 
             1)

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
                cv2.line(frame, 
                         (note_center_x - 10, note_center_y), 
                         (note_center_x + 10, note_center_y), 
                         (255, 0, 0), 2)
                cv2.line(frame, 
                         (note_center_x, note_center_y - 10), 
                         (note_center_x, note_center_y + 10), 
                         (255, 0, 0), 2)
            else:
                # Green box - Red crosshair
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.line(frame, 
                         (note_center_x - 10, note_center_y), 
                         (note_center_x + 10, note_center_y), 
                         (0, 0, 255), 2)
                cv2.line(frame, 
                         (note_center_x, note_center_y - 10), 
                         (note_center_x, note_center_y + 10), 
                         (0, 0, 255), 2)

            # Add metrics text to detected object
            # Text on top of detected object
            cv2.putText(frame, 
                        f'ID[{note["id"]}], W[{w}], H[{h}], A[{note["area"]}]', 
                        (x+5, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            # Text on bottom of detected object
            cv2.putText(frame, 
                        f'Range[{round(range * 39.37, 2)}in], Angle:[{angle_data[0]}/{angle_data[1]}deg]', 
                        (x+5, (y+h)+18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

    # Sort the array of found notes by the area, so the largest items are at the top
    detected_notes = sorted(detected_notes, key=lambda x: x['area'])

    return detected_notes

# Start the program
if __name__ == "__main__":
    # The video source camera identifier.  0 is generally the embedded webcam of most PC's, unless a second 
    # camera is attached.
    video_source = 0

    # Open the video capture
    cap = cv2.VideoCapture(video_source)
    
    # Increase the frames per second (FPS) returned from the camera
    cap.set(cv2.CAP_PROP_FPS, 20) 

    # Calculate the size and center of frame.  Only need to do this once and pass 
    # the details into the detect_orange_objects method.
    frame_details = capture_frame_details(cap)

    # Start an infinite loop to capture frames from teh camera, run the 
    # note detection algorithm, show the modified frame in a window, 
    # and check to see if the user hits the "q" key to terminate the loop.
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Run the note detection algorithm
        detected_notes = detect_orange_objects(video_source, frame, frame_details)

        # Display the frame with bounding boxes
        cv2.imshow("FRC 2024 Note Detection", frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
