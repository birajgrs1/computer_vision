import cv2
import time
import math
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm

# pycaw library
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


capture = cv2.VideoCapture(0)
previousTime = 0
detector = htm.HandDetector(detectionConf=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumeControl = interface.QueryInterface(IAudioEndpointVolume)
# volumeControl.GetMute()
# volumeControl.GetMasterVolumeLevel()
volRange = volumeControl.GetVolumeRange()
minVolume = volRange[0]
maxVolume = volRange[1]
vol = 0
volBar = 400  
volPercentage = 0 


while True:
    success, img = capture.read()
    img = detector.findHands(img)
    landmarkLists = detector.findPosition(img, draw=False)
    # print(landmarkLists) 
    if len(landmarkLists) != 0:
       print(landmarkLists[4], landmarkLists[8])
       
       x1, y1 = landmarkLists[4][1], landmarkLists[4][2]
       x2, y2 = landmarkLists[8][1], landmarkLists[8][2]
       
       cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
       
       
       cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
       cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
       cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3, cv2.FILLED)
       cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
       
       length = math.hypot(x2 - x1, y2 - y1)  
       
       # print(length)
       # Hand Range: 50 to 300
       # Volume Range: -65 to 0
       vol = np.interp(length, [50, 300], [minVolume, maxVolume])
       volBar = np.interp(length, [50, 300], [400, 150]) 
       volPercentage = np.interp(length, [50, 300], [0, 100])  # For percentage of volume

       print(int(length), vol)
       volumeControl.SetMasterVolumeLevel(vol, None)
       
       if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            
    # Drawing volume bar
    cv2.rectangle(img, (50, 150), (65, 400), (0, 255, 0), 3)  
    cv2.rectangle(img, (50, int(volBar)), (65, 400), (0, 255, 0), cv2.FILLED) 
    
    # Display volume percentage
    cv2.putText(img, f'{int(volPercentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    
        
       
    # Calculate the current time to compute FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    
    # Display FPS on the image
    cv2.putText(img, f"FPS: {int(fps)}", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 155, 0), 2)
    
    img = cv2.resize(img, (850, 450))
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
