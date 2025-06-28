import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

haar_cascade = cv.CascadeClassifier('assets/haar_face.xml')

people = ['Donald_Trump','Elon_Musk','Jeff_Bezos']

DIR = os.path.join(os.path.dirname(__file__),'assets','Training_Data')

features =[]
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)


            # if img_array is None:
            #     print(f"[WARN] Failed to read image: {img_path}")
            #     continue



            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)

            for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print('  Training done ----------------------')

features = np.array(features,dtype='object')
labels = np.array(labels)

# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the recognizer on features list and labels list

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)