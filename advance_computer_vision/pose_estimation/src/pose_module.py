import cv2
import time
import os
import mediapipe as mp


class poseDetector:
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth_landmarks,
                                     min_detection_confidence=self.detectionConf,
                                     min_tracking_confidence=self.trackConf)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    video_path = "D:/computer_vision/advance_computer_vision/pose_estimation/static/videos/jumping_rope.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: The video file was not found at {video_path}")
        return

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Could not open video file.")
        return

    previousTime = 0
    detector = poseDetector()

    while True:
        success, img = capture.read()
        if not success:
            print("Error: Could not read frame from video.")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)

        # Calculate FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
