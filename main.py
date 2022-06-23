import cv2
import numpy as np
import dlib 
from imutils import face_utils
import winsound
from scipy.spatial import distance as dist
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video1.mp4')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)
def compute(pointA, pointB):
    return np.linalg.norm(pointA - pointB)
def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2*down)
    if ratio > 0.25: return 2
    elif ratio>0.21: return 1
    else: return 0
def cal_yawn(shape): 
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = dist.euclidean(top_mean,low_mean)
    return distance
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(gray,face)
        landmarks = face_utils.shape_to_np(landmarks)
        left_blink = blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])
        yawn =cal_yawn(landmarks)
        if left_blink==0 or right_blink==0:
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep>8):
                status = "Drowsy/Sleeping"
                winsound.Beep(440, 900)
                color = (255,0,0)
        elif left_blink==1 or right_blink==1 or yawn > 35:
            sleep = 0
            active = 0
            drowsy+=1
            if (drowsy > 8):
                status = "Drowsy"
                winsound.Beep(440, 900)
                color = (255,255,0)
        else:
            drowsy = 0
            sleep = 0
            active +=1
            if (active>8):
                status = "Active"
                color = (0,255,0)
        #if status!="Active":
            #winsound.Beep(440, 500)
        cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_COMPLEX,1.2,color,3)
        for n in range (0,68):
            (x,y) = landmarks[n]
            cv2.circle(frame,(x,y),1,(255,0,0),-1)
    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(1)
    if key==27: break
