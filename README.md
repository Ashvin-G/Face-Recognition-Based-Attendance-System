# Face Recognition Based Attendance System
This project will facilitate institution and organization to conduct easy attendance and maintaining it. 
# Installation
This project deals with plethora of Image processing done by OpenCV with Python 3.7.4.
To install OpenCV for Python
```
$ pip install opencv-python
```
To install numpy
```
$ pip install numpy
```
In order to detect face you need haarcascade xml file seperately downloaded.
# Usage
After downloading the project navigate to scripts folder and follow the following steps.
1. Run the dataset.py script. This will capture your face through your primary Webcam and collect 20 sample face images and store it in dataset folder in grayscale mode.
2. Run the trainer.py script. This will train the captured 20 sample image into suitable matrix format and store it in trained_data folder named TrainingData.yml. This is followed by deleting the captured images from datasets folder to reduce redundancy.
3. Run detector.py script. This script will open up your primary Webcam and detect the face. Once the face is detected, along with user/student unique ID a timestamp is registered on Present_list.txt present in result folder. 
