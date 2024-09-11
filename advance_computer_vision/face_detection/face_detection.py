import cv2
import time
import mediapipe as mp

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)   # by default detection confidence value = 0.5 

capture = cv2.VideoCapture('static/videos/3.mp4')
previousTime = 0

# if not capture.isOpened():
#     print("Error: Could not open video.")
#     exit()

while True:
    success, img = capture.read()

    if not success:
        print("Finished processing the video.")
        break

    # converting bgr image to rgb 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            boundingBoxClass = detection.location_data.relative_bounding_box
            img_height, img_width, img_channels = img.shape
            boundingBox = (int(boundingBoxClass.xmin * img_width), int(boundingBoxClass.ymin * img_height), 
                           int(boundingBoxClass.width * img_width), int(boundingBoxClass.height * img_height))  # (x, y, width, height)

            cv2.rectangle(img, boundingBox, (255, 0, 254), 5)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (boundingBox[0], boundingBox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 3)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    img = cv2.resize(img, (640, 360))

    cv2.putText(img, f'fps: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 245, 0), 2)

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
