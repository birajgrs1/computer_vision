import cv2
import time 
import pose_module as pm 
import os

# Define the video path
video_path = "D:/computer_vision/advance_computer_vision/pose_estimation/static/videos/jumping_rope.mp4"

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: The video file was not found at {video_path}")
    exit()

# Initialize video capture
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    print("Error: Could not open video file.")
    exit()

previousTime = 0
detector = pm.poseDetector()  

while True:
    success, img = capture.read()
    if not success:
        print("Error: Could not read frame from video.")
        break

    # Find pose and landmarks
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    print(lmList)

    # Calculate FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
