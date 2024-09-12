import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
FACE_CONNECTIONS = mpFaceMesh.FACEMESH_TESSELATION


drawSpec = mpDraw.DrawingSpec(thickness =2 , circle_radius =2)

capture = cv2.VideoCapture('static/videos/2.mp4')
previousTime = 0

while True:
    success, img = capture.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, FACE_CONNECTIONS, drawSpec, drawSpec)
            
            for id, lms in enumerate(faceLandmarks.landmark):
                # print(lms)
                img_height, img_width, img_channel = img.shape
                
                x,y = int(lms.x*img_width), int(lms.y*img_height)
                print(id,x,y)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 6)

    img = cv2.resize(img, (640, 440))
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
