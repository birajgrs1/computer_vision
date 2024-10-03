import cv2  # OpenCV is used as cv2 in Python

# Reading the image
img1 = cv2.imread('./images/img1.jpg',1)  # 1 means colored ---> this image is bydefault (colored)
# Resizing the image to width 420 and height 240
img1 = cv2.resize(img1, (420, 240))
print(img1)
# Display the  image in a window
cv2.imshow("Original", img1)


# cv2: IMREAD_GRAYSCALE: loads image in grayscale mode
img2 = cv2.imread('./images/img2.jpg', 0)  # 0 for grayscale
img2 = cv2.resize(img2, (420, 240))
cv2.imshow("Gray Scale image", img2)
print("Image in grayscale === \n", img2)



# cv2.waitKey(3000)  # Holding image for 3second
cv2.waitKey(0)    # Controls the visualizations
cv2.destroyAllWindows()
