# Lane Curvature Detection Using Image Processing
This project is completed as a research internship under Computer Engineering Professorship at Technische Universität Chemnitz. It is a computer vision/image processing approach to detect lane curvature from highway video. Lane curvature detection from the highway is one of the fundamental requirements for both autonomous driving and advanced driver assistance (ADAS) based applications. Curvy lane is a crucial factor that plays a significant role in the increasing number of road fatalities worldwide. Detection of the curvature radius of the lane and providing a warning message for upcoming curvy road conditions have the potential to improve driving safety. Proposed algorithm has been implemented in Python3 and evaluated on a Raspberry Pi 3 B+ environment. It can detect and calculate lane curvature as well as provide warning messages for curvy road conditions.

### Project Goal
Goals of this project are as follows:
- Detect lane from a video captured from the highway
- Detect and measure lane curvature
- Provide warning message in terms of curvy lane condition
- Evaluate the proposed method on a Raspberry Pi 3 B+ environment

### Hardware and Software requirements
- Raspberry Pi
  The ultimate goal of this proposed method of lane curvature detection is to use it
for an in-vehicle system. Raspberry Pi suits perfectly to serve the purpose as target
hardware to run and test the method. The model that has been chosen is Rasp-
berry Pi 3 B+ which is a Linux based single-chip computer. It is also possible to run on any other machine.
- Python Programming Language
- OpenCV
- NumPy

### Project Architecture
Assets folder contains two input video file to test the algorithm. Both lanecurvaturedetection.py and lanecurvaturedetection.ipynb have the same functionality. Use lanecurvaturedetection.ipynb to run from Jupyter Notebook. Use lanecurvaturedetection.py to run it from python IDE, windows command prompt or Raspberry Pi terminal. Make sure the assets folder is in the same directory.
