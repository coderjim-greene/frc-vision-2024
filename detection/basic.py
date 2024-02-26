"""
File: basic_webcam_example.py
Author: Jim Greene
Date: 3-Feb-2023

Description: Simple OpenCV application which displays a video stream from a web camera, 
             until the user presses "q" on the keyboard to exit.
"""

import cv2 

# Use OpenCV to capture video from the camera
cap = cv2.VideoCapture(0)

# Run an infinite loop to:
#     a) read frames from the camera using cap.read()
#     b) display the images in a window with the title "Note Detection"
#     c) Check to see if the user presses "q", which will terminate the loop
while True:
    ret, frame = cap.read()
    cv2.imshow('Note Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the link to the webcam
cap.release()

# Destroy the window used to show the images
cv2.destroyAllWindows()