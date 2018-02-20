__author__ = 'gerry'

import cv2





#lower_body_cascade
lower_body_cascade=cv2.CascadeClassifier("haarcascades/haarcascade_lowerbody.xml")

#read image as bw which is more accurate
#https://github.com/opencv/opencv/tree/master/data/haarcascades

#orig face photo
#img=cv2.imread("photo.jpg")



#setyup the face cascade
#faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)

#lowerbody
#lower_body=lower_body_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)



#draw rectangle after detection





#for x,y,w,h in lower_body:
#    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)






#detect yes in the image file
def detecteyes( imgFile ):

    img=cv2.imread(imgFile)

    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #setup haarcascade
    eye_cascade=cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

    #eyes
    eyes=eye_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)

    #draw rectangle
    for x,y,w,h in eyes:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    #resizeimg with correct aspect ratio
    resized_img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

    cv2.imshow("grey",resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectfaces( imgFile ):

    img=cv2.imread(imgFile)

    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #haarcascade xml file
    face_cascade=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

    #eyes
    faces= face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)

    #draw rectangle
    for x,y,w,h in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    #resizeimg with correct aspect ratio
    resized_img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

    cv2.imshow("grey",resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detecteyes("images/trump_sons.jpg")

detectfaces("images/trump_sons.jpg")