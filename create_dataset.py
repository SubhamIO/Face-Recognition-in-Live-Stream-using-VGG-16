import cv2
import numpy as np
#we will try to detect the face of individuals using the haarcascade_frontalface_default.xml
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    #Now the 2nd step is to load the image and convert it into gray-scale.
    '''Generally the images that we see are in the form of RGB channel(Red, Green, Blue). So, when OpenCV reads the RGB image,
     it usually stores the image in BGR (Blue, Green, Red) channel. For the purposes of image recognition, we need to convert this BGR channel to gray channel.
     The reason for this is gray channel is easy to process and is computationally less intensive as it contains only 1-channel of black-white.'''
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    '''Now after converting the image from RGB to Gray, we will now try to locate the exact features in our face.
    This detectMultiScale() function will help us to find the features/locations of the new image.
    The way it does is, it will use all the features from the face_classifier object to detect the features of the new image.'''

    '''Parameters for detectMultiScale(gray scale variable,scaleFactor,minNeighbors)'''
    '''scaleFactor = Parameter specifying how much the image size is reduced
                    at each image scale.

       minNeighbors = Parameter specifying how many neighbors each candidate rectangle should have to retain it.
                    This parameter will affect the quality of the detected faces. Higher value results in fewer detections but with higher quality.
                    3~6 is a good value for it. In our case, I have taken 5 as the minNeighbors and this has worked perfectly for the image that I have used.'''

    '''detectMultiScale returns 4 values — x-coordinate, y-coordinate, width(w) and height(h) of the detected feature of the face.
                        Based on these 4 values we will draw a rectangle around the face.'''
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_faces=img[y:y+h,x:x+w]


    return cropped_faces



cap=cv2.VideoCapture(0)
count=0

while True:
    ret,frame=cap.read()
    '''Extract face , convert to grayscale and save it in out folders'''
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path='/Users/subham/Desktop/Recognition Using Cnn/data/user'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count==200:
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!!")
