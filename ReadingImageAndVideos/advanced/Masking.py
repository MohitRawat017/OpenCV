# The intuition behind masking is to prevent certain parts of the image/input from being processed so that the model focuses on relevant features.
import cv2 as cv 
import numpy as np

img = cv.imread('../image.png')
cv.imshow('Input Image', img)

# Create a mask with the same dimensions as the image, initialized to zeros (black)
blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Circle Mask', circle)
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
cv.imshow('Rectangle Mask', rectangle)
# Apply the circle mask to the image using bitwise AND operation
masked_circle = cv.bitwise_and(src1 =img, src2=img, mask=circle)
cv.imshow('Masked Circle', masked_circle)

# Apply the rectangle mask to the image using bitwise AND operation
masked_rectangle = cv.bitwise_and(src1 =img, src2=img, mask=rectangle)
cv.imshow('Masked Rectangle', masked_rectangle)

cv.waitKey(0)
cv.destroyAllWindows()
