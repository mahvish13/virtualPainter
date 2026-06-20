import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


brushThickness = 25
eraserThickness = 100
strokes=[]



folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65,maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros(( 720,1280, 3), np.uint8)
clearFlag = False


while True:

    # 1. Import image
    success, img = cap.read()
    if not success or img is None:
        cv2.waitKey(50)
        continue
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img, draw=False)

    if len(lmList) !=0:
        # Distance between thumb and index finger
        length, img, _ = detector.findDistance(4, 8, img)

        # Dynamic brush size
        brushThickness = int(np.interp(length, [20, 200], [5, 50]))

        # Dynamic eraser size
        eraserThickness = int(np.interp(length, [20, 200], [20, 100]))

        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # Clear canvas when all thumb and index up;

        if fingers == [1, 0, 0, 0, 1]:
            if not clearFlag:
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                xp, yp = 0, 0
                clearFlag = True
        else:
            clearFlag = False


        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1]==1 and fingers[2] == 0:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if abs(x1 - xp) > 5 or abs(y1 - yp) > 5:
                strokes.append((xp, yp, x1, y1, drawColor, brushThickness))
            print("Drawing Mode")


            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness,cv2.LINE_AA)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness,cv2.LINE_AA)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness,cv2.LINE_AA)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness,cv2.LINE_AA)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness,cv2.LINE_AA)

            xp, yp = x1, y1


        # # Clear Canvas when all fingers are up
        # if all (x >= 1 for x in fingers):
        #     imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # Setting the header image

    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    key = cv2.waitKey(1) & 0xFF
    # Press 'c' to clean drawing
    if key == ord('c'):
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

        _, imgTh = cv2.threshold(imgBlur, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        imgClean = cv2.morphologyEx(imgTh, cv2.MORPH_CLOSE, kernel, iterations=2)

        imgClean = cv2.dilate(imgClean, kernel, iterations=1)

        cv2.imshow("Clean Drawing", imgClean)

    # Press 's' to save drawing
    if key == ord('s'):
        cv2.imwrite(f"drawing_{int(time.time())}.png", imgCanvas)
        print("Image Saved!")

    # Press 'q' to quit
    if key == ord('q'):
        break
    # Undo last stroke
    if key == ord('u'):
        if strokes:
            strokes.pop()

        # Clear canvas
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        # Redraw all strokes except last
        for s in strokes:
            cv2.line(imgCanvas, (s[0], s[1]), (s[2], s[3]), s[4], s[5],cv2.LINE_AA)
cap.release()
cv2.destroyAllWindows()
