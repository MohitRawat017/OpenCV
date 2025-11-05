import cv2 as cv

img = cv.imread('../image.png')
cv.imshow('Original Image', img)
# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale Image', gray)

# Simple Thresholding
threshold,thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# baiscally looks at each pixel and if the value is greater than 127, it is set to 255 (white), otherwise it is set to 0 (black)
cv.imshow('Simple Thresholding', thresh)

# inverse binary thresholding
threshold,thresh_inv = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse Binary Thresholding', thresh_inv)

# Adaptive Thresholding
# this method finds the optimal threshold for smaller regions of the image
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY, blockSize=11, C=2)

cv.imshow('Adaptive Thresholding', adaptive_thresh)
cv.waitKey(0)
cv.destroyAllWindows()
