import cv2  
import mediapipe as mp 
import time  
import handtrackingModule as htm


previousTime = 0
currentTime = 0
capture = cv2.VideoCapture(0)

detector = htm.HandDetector()  # instantiate

while True:
    success, img = capture.read()
    img = detector.findHands(img)
    lm_list = detector.findPosition(img)

    if lm_list:
        print(lm_list[4])

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
