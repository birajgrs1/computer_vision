import cv2
import time
import mediapipe as mp


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon) 
        
    def findFaces(self, img, draw=True):
        # converting bgr image to rgb 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        boundingBoxes = []
       
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                '''
                mpDraw.draw_detection(img, detection)
                print(id, detection)
                print(detection.score)
                print(detection.location_data.relative_bounding_box)
                '''
                boundingBoxClass = detection.location_data.relative_bounding_box
                img_height, img_width, img_channels = img.shape
                boundingBox = (int(boundingBoxClass.xmin * img_width), int(boundingBoxClass.ymin * img_height), 
                               int(boundingBoxClass.width * img_width), int(boundingBoxClass.height * img_height))  # (x, y, width, height)
                
                boundingBoxes.append((id, boundingBox, detection.score))
                
                if draw:
                    self.fancyDraw(img, boundingBox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (boundingBox[0], boundingBox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 3)
        
        return img, boundingBoxes
    
    def fancyDraw(self, img, boundingBox, length=30, thickness=10, rectThickness=1):
        x, y, width, height = boundingBox
        x1, y1 = x + width, y + height
        
        # Drawing the rectangle with a specified thickness
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 254), rectThickness)
        
        # Top thickness
        cv2.line(img, (x, y), (x + length, y), (255, 0, 255), thickness)
        
        # Bottom thickness
        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        
        # Left thickness
        cv2.line(img, (x, y), (x, y + length), (255, 0, 255), thickness)
        
        # Right thickness
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness)
        
        return img

def main():
    capture = cv2.VideoCapture('static/videos/3.mp4')
    previousTime = 0
    detector = FaceDetector()

    while True:
        success, img = capture.read()
        if not success:
            print("Finished processing the video.")
            break

        img, _ = detector.findFaces(img)

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

if __name__ == '__main__':
    main()
