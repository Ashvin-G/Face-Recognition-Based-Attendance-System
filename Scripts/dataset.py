import cv2
import numpy as np

from tkinter import *
from functools import partial

def validateLogin(username, password):
	print("username entered :", username.get())
	print("password entered :", password.get())
	return

#window
tkWindow = Tk()  
tkWindow.geometry('400x150')  
tkWindow.title('Tkinter Login Form - pythonexamples.org')

#username label and text entry box
usernameLabel = Label(tkWindow, text="User Name").grid(row=0, column=0)
username = StringVar()
usernameEntry = Entry(tkWindow, textvariable=username).grid(row=0, column=1)  

#password label and password entry box
passwordLabel = Label(tkWindow,text="Password").grid(row=1, column=0)  
password = StringVar()
passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=1, column=1)  

validateLogin = partial(validateLogin, username, password)

#login button
loginButton = Button(tkWindow, text="Login", command=validateLogin).grid(row=4, column=0)  

tkWindow.mainloop()
faces = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

Id = input("Enter User ID : ")
sample_num = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = faces.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        sample_num = sample_num + 1
        cv2.imwrite("../datasets/user"+"."+str(Id)+"."+str(sample_num)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        cv2.waitKey(10)

    
    if sample_num > 20:
        break
    

cap.release()
cv2.destroyAllWindows()
