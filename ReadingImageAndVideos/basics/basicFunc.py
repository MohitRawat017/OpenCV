import cv2 as cv

img = cv.imread("image.png")
cv.imshow("image", img)
cv.waitKey(0)
# Converting image to Greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
cv.waitKey(0)

# Blur the image 
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
# (7,7) is the kernel size, it should be odd numbers 
# higher the kernel size, more the blur 
cv.imshow("Blur", blur)
cv.waitKey(0)

# Edge Cascade : basically detects the edges in the image
# cv.Canny(image, threshold1, threshold2)
# these thresholds are for the hysteresis procedure 
# lower threshold is for edge linking 
# upper threshold is for finding initial segments of strong edges
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny", canny)
cv.waitKey(0)


# Dilating the image : basically thickens the edges
# cv.dilate(image, kernel, iterations)
# kernel is a matrix that decides how much to dilate
# iterations is how many times to apply the dilation 
# generally the higher the iterations, the thicker the edges
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow("Dilated", dilated)
cv.waitKey(0)

# Eroding the image : basically thins the edges
# cv.erode(image, kernel, iterations)
eroded = cv.erode(dilated, (7,7), iterations = 3)
cv.imshow("Eroded", eroded)
cv.waitKey(0)

# Resize the image 
# cv.resize(image, (width, height))
resized = cv.resize(img, (500,500))
cv.imshow("Resized", resized)
cv.waitKey(0)

# Cropping the image
# img[y1:y2, x1:x2]
cropped = img[50:200, 200:400]
cv.imshow("Cropped", cropped)
cv.waitKey(0)

cv.destroyAllWindows()
