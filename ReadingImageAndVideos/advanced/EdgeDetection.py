import cv2 as cv 
import numpy as np

img = cv.imread('../image.png')
cv.imshow('Original Image', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale Image', gray)

# Laplacian Edge Detection
Laplacian = cv.Laplacian(gray, cv.CV_64F)
Laplacian = np.uint8(np.absolute(Laplacian))
cv.imshow('Laplacian Edge Detection', Laplacian)

# Sobel Edge Detection
sobelX = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobelY = cv.Sobel(gray, cv.CV_64F, 0, 1)
cv.imshow('Sobel X', sobelX)
cv.imshow('Sobel Y', sobelY)
combinedSobel = cv.bitwise_or(sobelX, sobelY)
cv.imshow('Combined Sobel', combinedSobel)

# Canny Edge Detection
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny Edge Detection', canny)


cv.waitKey(0)
cv.destroyAllWindows()