import cv2
import numpy as np
import os
from PIL import Image

datadir = '..\\datasets'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("..\\haarcascade\\haarcascade_frontalface_default.xml");

def get_img_and_id(datadir):
    img_paths = [os.path.join(datadir, img_path) for img_path in os.listdir(datadir)]

    img_list = []
    ids_list = []

    for img_path in img_paths:
        pilImage = Image.open(img_path).convert('L')
        image_np = np.array(pilImage,'uint8')

        Id = int(os.path.split(img_path)[-1].split(".")[1])

        faces = detector.detectMultiScale(image_np)

        for (x, y, w, h) in faces:
            img_list.append(image_np[y:y+h, x:x+w])
            ids_list.append(Id)

    return img_list, ids_list
        
        

faces, Ids = get_img_and_id(datadir)
recognizer.train(faces, np.array(Ids))
recognizer.save('..//trained_data//TrainingData.yml')

for img_path in os.listdir(datadir):
    path = os.path.join(datadir, img_path)
    os.remove(path)
