import cv2 as cv

img = cv.imread('../image.png')
# Average Blurring
# cv.blur(image , kernel_size)
blurred = cv.blur(img, (7, 7))
cv.imshow("Blurred Image", blurred)

# Gaussian Blurring
# cv.GaussianBlur(image , kernel_size , sigmaX)
# sigmaX or sigmaY basically defines how much the image is blurred.
# A higher value means more blurring.
gaussian_blurred = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow("Gaussian Blurred Image", gaussian_blurred)

# Median Blurring 
# cv.medianBlur(image , kernel_size)
median_blurred = cv.medianBlur(img, 7)
# median blur only takes a single odd integer value . it is basically (7,7) only.
cv.imshow("Median Blurred Image", median_blurred)

# Bilateral Blurring -> Best for preserving edges
# cv.bilateralFilter(image , diameter , sigmaColor , sigmaSpace)
# sigmaColor -> how much the colors (intensity) should be considered while blurring
# sigmaSpace -> how much the space (proximity) should be considered while blurring
bilateral_blurred = cv.bilateralFilter(img, 9, 100, 100)
cv.imshow("Bilateral Blurred Image", bilateral_blurred)

# lets check the contours for all the blurred images
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 125, 175)
canny_blurred = cv.Canny(bilateral_blurred, 125, 175)
canny_gaussian = cv.Canny(gaussian_blurred, 125, 175)
cv.imshow("Canny Edges on Original Image", canny)
cv.imshow("Canny Edges on Bilateral Blurred Image", canny_blurred)
cv.imshow("Canny Edges on Gaussian Blurred Image", canny_gaussian)


cv.waitKey(0)
cv.destroyAllWindows()

# why do we use bilateral_blurring ?
# Because it preserves the edges while blurring the image.
# In other blurring techniques, edges tend to get blurred as well.
# In bilateral blurring, the filter considers both the spatial proximity and the intensity difference,
# allowing it to smooth regions while maintaining sharp edges.