import cv2
import time
import os
import mediapipe as mp

# Initialize MediaPipe Pose model
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

video_path = "D:/computer_vision/advance_computer_vision/pose_estimation/static/videos/jumping_rope.mp4"

if not os.path.exists(video_path):
    print(f"Error: The video file was not found at {video_path}")
else:
    # Initialize video capture
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print("Error: Could not open video file.")
    else:
        previousTime = time.time()

        while True:
            success, img = capture.read()
            if not success:
                print("Error: Unable to read frame or end of video.")
                break

            # Convert the image to RGB (required for MediaPipe)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                for id, lm in enumerate(results.pose_landmarks.landmark):
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    # print(id,lm)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Image", img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
