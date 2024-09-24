import cv2
import os
import time
from HandTrackingModule import HandDetector

wCam, hCam = 640, 580
capture = cv2.VideoCapture(0)
previousTime = 0
capture.set(3, wCam)
capture.set(4, hCam)

folderPath = r'D:\computer_vision\advance_computer_vision\project-2_FingerCounter\static\images'
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imgPath in myList:
    fullPath = os.path.join(folderPath, imgPath)
    image = cv2.imread(fullPath)
    overlayList.append(image)

detector = HandDetector(detectionConf=0.7)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = capture.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb detection
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers detection
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        if totalFingers <= len(overlayList) and totalFingers > 0:
            overlay = cv2.resize(overlayList[totalFingers - 1], (200, 200))
            img[0:200, 0:200] = overlay

        # Draw rectangle to show the count of fingers
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # Calculate FPS and display on the image
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('image', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
