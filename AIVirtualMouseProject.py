import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy #allow us to move around with our mouse

##########################
wCam, hCam = 640, 480#setting width  off camera )#setting height off camera
frameR = 100  # Frame Reduction  for kitne area me mouse detect kre
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0 #previous location of x and y and both are varible
clocX, clocY = 0, 0 #current location of x and y and both are varible

cap = cv2.VideoCapture(0)#to run our webcam
cap.set(3, wCam)#setting width  off camera
cap.set(4, hCam)#setting height off camera
detector = htm.handDetector(maxHands=1)#object of the detector class from handtracking module
wScr, hScr = autopy.screen.size()#width of scrren and height of screen to convert cordinated according to ours
# print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)#yeh return krega previous funciton se
    lmList, bbox = detector.findPosition(img)# ye bhi kra h pahle    #bbox mtl bounding box joisme apna hath aa rha h
    # 2. Get the tip of the index and middle fingers
    #yadi keval index ka tip uper rhega toh cursor move karega
    #but if dono finger ke tip uper rhae and dono ke bich ki distance ek certain distancese kam h toh click kam kregs
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]#for storing landmarks of index finger or we can say that dimesniosn in (x,y)
        x2, y2 = lmList[12][1:]#similar then above
        # print(x1, y1, x2, y2)

    # 3. Check which fingers are up
    fingers = detector.fingersUp()#isme array aa rha 5 ungli ki value aati h jo up h voh
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    # 4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        # why it is needed to convert accordint to our screen
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6. Smoothen Values
        # it will  not vary and not flicker around
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Move Mouse  #move our mouse according to that

        autopy.mouse.move(wScr - clocX, clocY)#ye move krega mouse with our current convertd cordinates
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)#for acha lgepata chaleki mouse moov ho rha
        plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)#distnce bw both fingers tip
        print(length)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255,
                              0    ,0), cv2.FILLED)
            autopy.mouse.click()

    # 11. Frame Rate is for console
    #yeh bhi simple h handtracking wale me kara tha
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display simple ever
    cv2.imshow("Image", img)
    cv2.waitKey(1)