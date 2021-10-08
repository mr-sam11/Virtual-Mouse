

import cv2
import mediapipe as mp
import time  #check fro framerate
import math
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):#this is alll are iniitalization of object
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #this we can say that formality for using thesse module
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)#object of hands class eith different parameters

        #we can easily understand prameters of that above object

        self.mpDraw = mp.solutions.drawing_utils#we have this method provideed by mediapipe that actually helps us to draw all the points (total21points are there)
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):#for drawing hands only
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#we pass our img for converting in rgbrgb image
        self.results = self.hands.process(imgRGB)#the .process is method inside the object hand and will create fram for us
        # print(results.multi_hand_landmarks)#this is to check that anything is detected or not
        #if we not putting our hand then it will display none or if we show our hands then it will print frame rate or dimension

        #now we want to access the information whcih the above rsult object have
        # as the parameter we can use multihand so .......
        if self.results.multi_hand_landmarks:#this if condition check that there is something detedted then it will go down n if block
            for handLms in self.results.multi_hand_landmarks:#for each handmaks in result
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                    self.mpHands.HAND_CONNECTIONS)#we have this method provideed by mediapipe that actually helps us to draw all the points (total21points are there)
        # (mpDraw)we are using these as a object of that method
        #HAND_CONNECTIONS this is for drawa the connection (line jo aa rhi h )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:#check tha5 htere we are detecting any landmark or not
            #then travers whole loop for find position
            #this method for only one perticula hand
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):#this landmarke what we are getting and id is which is related to the exact id of landmark
                # print(id, lm) for evry iteration we have one id and for evry id we have 3 landmarks (X,Y,Z) (dimension in 3d plane
                #this landmark are in deimal and bhut badi value h jo ki ration h image ka
                #hamari image ya jo camera dikh rha uski toh height idth kam h
                #so ab landmarke ko deimal me lane ke liyewe are multplying wiht widht and height

                h, w, c = img.shape#this is the function and give us widhtand height
                cx, cy = int(lm.x * w), int(lm.y * h)#position of the center in demial
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:#this is for all landmark drawingthe circle with differnet size and diff colour
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
            (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0#variable for storing previous time
    cTime = 0#variable for storing current  time
    cap = cv2.VideoCapture(0) # video object
    detector = handDetector()#object of hand detector class
    while True:
        success, img = cap.read()#that will give us our frame
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()#this will give us current time
        fps = 1 / (cTime - pTime)#fps i think frame rate ration
        pTime = cTime# previous time become current time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)#for displayin on image rather than display on console

        cv2.imshow("Image", img)# thisis wht to display the web cam
        cv2.waitKey(1)#wait till we close

if __name__ == "__main__":
    main()