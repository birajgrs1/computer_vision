import cv2  
import mediapipe as mp 
import time  

capture = cv2.VideoCapture(0)

# Initialize MediaPipe Hands for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False)
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = capture.read()
    
    # Check if the frame was successfully captured
    if not success:
        print("Failed to grab frame")
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                # Get image dimensions
                height, width, channelsOfImg = img.shape
                # Convert normalized landmark coordinates to pixel coordinates
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                print(id, cx, cy)

                # if id == 5:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 254), cv2.FILLED)
            
            # Draw hand landmarks and connections on the image
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)
    
    # Calculate the current time to compute FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Display FPS on the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # Show the image with landmarks and FPS
    cv2.imshow("Image", img)
    
    # Wait for 1 millisecond and check if the 'q' key was pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
