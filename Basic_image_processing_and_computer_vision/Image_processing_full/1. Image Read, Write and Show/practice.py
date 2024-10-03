import cv2  

# Image conversion ---> colored image into grayscale.
path = input("Enter the path and name of the image === ")
print("You entered this ===", path)

img = cv2.imread(path, 0)  # Load the image in grayscale mode

# img2 = cv2.resize(img, (840, 560))

# img = cv2.flip(img, 0)  # Accepts 0, 1, or -1 as flipping options

# Display the image
cv2.imshow("Converted image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
