import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

data_path ='C:/Users/Tech/pythonProject2/open cv/'

onlyfiles = [f for f in listdir(data_path)if isfile(join(data_path,f))] #new method write the python  for loop

Training_Data, Labels = [],[]# saved image called

for i, files in enumerate(onlyfiles):# data called
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    #unsigned integer 8th bits
    Labels.append (i)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create() #mode is calling build model
model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Model Training Complete!!!")
face_classifier = cv2.CascadeClassifier('C:/Users/Tech/pythonProject2/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#face_classifier = cv2.CascadeClassifier('C:/Users/Tech/pythonProject2/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return img,[]# roi=reasion of intrest
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    #phage to detect to frame

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1]<500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'% shivam jaiswal'
            cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)#text create now then check the face value by confidence  vareable
            if confidence > 75:
                cv2.putText(image, "unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)# position change below side and color haveing green unlocked
                cv2.imshow('Face Cropper',image)
            else:
                cv2.putText(image, "locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),# locked
                            2)  # if image not match the work this
                cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face not found ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),
                    2)
       # cv2.putText(image, "Loked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0 , 255), 2)
        cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1)==13:
        break
        cap.release()
        cv2.destroyALLWindows()














































