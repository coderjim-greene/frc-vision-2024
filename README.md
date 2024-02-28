# Image Processing Examples

## Setup Instructions
### 1.  Create a Virtual Environment.
Create a new directory, and within it create a Python Virtual Environment.  The .gitignore file for this project already ignores the directory name .venv.  Then source the virtual environment's ```activate``` script.

```
python3 -m venv .venv
source .venv/bin/activate
```

### 2.  Clone This Repository
In the same directory, clone the repository using the following git command.

```
git clone https://github.com/coderjim-greene/frc-vision-2024.git
```

### 3.  Install Application Dependencies
Enter the following command to install the applications dependencies.

```
pip3 install -r requirements.txt
```

---


# About The Examples
These examples were built and tested using a Microsoft Lifecam HD 3000 camera.  If you are using a different type of webcam while testing this code, you will need to perform calibration of your camera to obtain values to make things like distance and pose estimation work correctly.

Disclaimer:  I am by no means an OpenCV or Python expert.  These are just some scripts that I managed to hack together over a few evenings, with some generous help from Google and ChatGPT.

## Calibration Information
The webcam used to develop these examples was calibrated using the scripts located in the ```/calibration``` directory.  Running ```/calibration/calibrate_camera.py``` uses images in the ```/calibration/calibration_images``` directory to calculate values used by OpenCV for the AprilTag example.


## Detection Examples
In the ```/detection``` folder, you will scripts demonstrating detection of orange toroid rings and AprilTags.  

The example scripts are:

- ```apriltag_pose_detection```
This script demonstrates detecting Apriltag objects in a video stream from the webcam, and projecting the pose of the tags (essentially the direction the tags face relative to the camera) onto the video.

- ```detect_notes_st.py```
This is a single-threaded implementation of detecting orange toroid objects, calculating their distance from the camera, as well as an angular estimation of the center of the detected object from the center of the image.  Bounding boxes are drawn around the detected objects, and visual changes to teh bounding box are implemented when the bounding box is located near the x-center of the image.

- ```detect_notes_mt.py```
This is a multi-threaded implementation of ```detect_notes_st.py```.  It was implemented as a test to see if multi-threading would dramatically improve the performance of detecting the orange toroid objects.  At the end of the day, the performance was fairly similar, and the complexity of the multi-threading code wasn't really worth the effort.




