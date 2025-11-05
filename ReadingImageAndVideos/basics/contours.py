# contours are not just edges, they are closed curves that define shapes in an image.
import cv2 as cv
import numpy as np

# Load the image and convert it to grayscale
img = cv.imread("image.png")
cv.imshow("Original Image", img) #type: ignore
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #type: ignore
cv.imshow("Grayscale Image", gray)

# lets blur the image to reduce noise and improve contour detection
blurred = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# what cv.BORDER_DEFAULT does is it applies a default border type when the kernel overlaps the image border during convolution.
cv.imshow("Blurred Image", blurred)

# Now we can use the Canny edge detector to find edges in the image
canny = cv.Canny(gray, 125, 175)
# so the minimum threshold is 125 and the maximum threshold is 175.
# meaning that any gradient value below 125 will be considered as non-edge (0), and any value above 175 will be considered as a strong edge (255).
cv.imshow("Canny Edges", canny)

canny2 = cv.Canny(blurred, 125, 175)
cv.imshow("Canny Edges on Blurred Image", canny2)

# Now we can find the contours in the edged image 
contours , heirarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# cv.RETR_LIST retrieves all of the contours without establishing any hierarchical relationships between them.
# the other options can be :
# cv.RETR_EXTERNAL retrieves only the extreme outer contours.
# cv.RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours.
# cv.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
# this helps to save memory and reduce the number of points stored for each contour.
print(f"Number of contours found: {len(contours)}")
# we used len() because contours is a list of all the contours found in the image.

# Now we can draw the contours on a blank image
blank = np.zeros(img.shape, dtype="uint8")
cv.drawContours(blank, contours, -1 , (255,0,0), 1)
# -1 means we want to draw all the contours.
cv.imshow("Contours Drawn", blank)
cv.waitKey(0)

# there is also the concept of thresholding in image processing.
# Thresholding is a technique used to segment an image by converting it to a binary image based on a specified threshold value.
ret, thresh = cv.threshold(gray, 125,255, cv.THRESH_BINARY)
cv.imshow("Thresholded Image", thresh)

# Now we can find the contours in the thresholded image
contours2, heirarchy2 = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f"Number of contours found in thresholded image: {len(contours2)}")
# Now we can draw the contours on a blank image
blank2 = np.zeros(img.shape, dtype="uint8")
cv.drawContours(blank2, contours2, -1 , (0,255,0), 1)
cv.imshow("Contours from Thresholded Image", blank2)
cv.waitKey(0)
# so in summary, contours are closed curves that define shapes in an image.
# we can find contours using edge detection or thresholding techniques.
# we can then draw these contours on a blank image for visualization.
# Contours are useful in various computer vision tasks such as shape analysis, object detection, and image segmentation.
# we can use the Canny edge detector or thresholding techniques to find contours in an image.

cv.destroyAllWindows()