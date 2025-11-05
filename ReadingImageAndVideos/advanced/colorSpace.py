import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("../image.png")
cv.imshow("Original", img)

# Convert BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", gray)

# BGR To HSV -> Hue, Saturation, Value
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)

# BGR To LAB -> Lightness, A channel, B channel
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
cv.imshow("LAB", lab)

# BGR To RGB
# Note: OpenCV uses BGR by default
# however, many libraries (like Matplotlib) use RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB", rgb)
plt.imshow(rgb)
plt.axis("off")  # Hide axis
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

# Note: there is no direct conversion from like:
# Grayscale to HSV or LAB in OpenCV.
# You must first convert Grayscale to BGR, then to the desired color space.