import cv2
import mediapipe as mp
import time


class FaceMeshDectector():
    
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.FACE_CONNECTIONS = self.mpFaceMesh.FACEMESH_TESSELATION
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)
        
    def findFaceMesh(self, img, draw=True):       
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for faceLandmarks in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                
                face = []
                for id, lms in enumerate(faceLandmarks.landmark):
                    img_height, img_width, img_channel = img.shape
                    
                    x, y = int(lms.x * img_width), int(lms.y * img_height)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

                    face.append((x, y))
                
                faces.append(face) 
                
        return img, faces


def main():
    capture = cv2.VideoCapture('static/videos/3.mp4')
    previousTime = 0
    
    detector = FaceMeshDectector()
    
    while True:
        success, img = capture.read()
        img, faces = detector.findFaceMesh(img, draw=True)
        
        if len(faces) != 0:
            print(f"Number of faces detected: {len(faces)}")
        
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

if __name__ == "__main__":
    main()
