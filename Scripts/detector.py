import cv2
import numpy as np
import os
from datetime import datetime

today = datetime.now()
date_time = today.strftime("%d/%m/%Y  %H:%M:%S")

face_cascade = cv2.CascadeClassifier('..//haarcascade//haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('..//trained_data//TrainingData.yml')

id = 0

file = open("..//result//Present_list.txt", "a")

present_list = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, configuration = recognizer.predict(gray[y:y+h, x:x+w])

        
        
        if id < 10 and id>=1:
            usn = "2GI17CS00" + str(id)
        if id < 99 and id>=10:
            usn = "2GI17CS0" + str(id)
        else:
            usn = "2GI17CS" + str(id)

        
        if usn not in present_list:
            present_list.append(usn)

        
        cv2.putText(frame, usn, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print(present_list)

for student in present_list:
    file.write(str(student)+"\t"+str(date_time)+"\n")
file.close()




