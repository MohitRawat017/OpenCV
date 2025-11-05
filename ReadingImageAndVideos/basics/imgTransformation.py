import cv2 as cv
import numpy as np

img = cv.imread("image.png")

cv.imshow("Image", img)

# Translation : shifting the image
def translate(img, x, y):
    transMatrix = np.float32([[1, 0, x], [0, 1, y]])

    dimensions = (img.shape[1], img.shape[0])
    # because OpenCV uses (width, height) format for image dimensions
    # cv.warpAffine(image, transformation matrix, output dimensions)
    return cv.warpAffine(img, transMatrix, dimensions)

# -x --> left   x --> right
# -y --> up   y --> down
translated = translate(img, 100, 100)
cv.imshow("Translated", translated)

# Rotation of image
def rotate(img, angle, rotationPoint=None):
    (height, width) = img.shape[:2]
    if rotationPoint is None:
        rotationPoint = (width // 2, height // 2) 
        # center of the image for rotation
    rotMatrix = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMatrix, dimensions)
rotated = rotate(img, 45)
cv.imshow("Rotated", rotated)

# Resizing the image
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)
# INTER_AREA for shrinking
# INTER_CUBIC and INTER_LINEAR for enlarging

# Flipping the image
# 0 --> vertical flip
# 1 --> horizontal flip
# -1 --> both
flipped = cv.flip(img, -1)
cv.imshow("Flipped", flipped)

# Cropping the image
cropped = img[200:400, 300:400]
cv.imshow("Cropped", cropped)
cv.waitKey(0)

cv.destroyAllWindows()