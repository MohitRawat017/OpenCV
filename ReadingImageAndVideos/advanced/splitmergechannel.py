import cv2 as cv 
import numpy as np 

# Load an image
img = cv.imread("../image.png")

# Split the image into its B, G, R channels
b, g, r = cv.split(img)

blue = cv.imshow("Blue Channel", b)
green = cv.imshow("Green Channel", g)
red = cv.imshow("Red Channel", r)
# image shape -> (height, width, channels)
# img -> (512, 512, 3)
# b -> (512, 512)
# g -> (512, 512)   
# r -> (512, 512)
# oh shouldn't it be (512, 512, 1)? No, because when we split the channels, we get 2D arrays for each channel.

# if we want to merge them back
merged = cv.merge([b, g, r])
cv.imshow("Merged Image", merged)

# however these blue, green and red will be gray scale images
# to visualize them in color we can create blank channels

blank = np.zeros(img.shape[:2], dtype="uint8")
blue_img = cv.merge([b, blank, blank])
green_img = cv.merge([blank, g, blank])
red_img = cv.merge([blank, blank, r])
cv.imshow("Blue Image", blue_img)
cv.imshow("Green Image", green_img)
cv.imshow("Red Image", red_img)

# merged blank bgr 
merged_blank = cv.merge([blank, blank, blank])
cv.imshow("Merged Blank Image", merged_blank)

cv.waitKey(0)
cv.destroyAllWindows()