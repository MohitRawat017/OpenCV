import cv2 as cv 
import numpy as np

# img = cv.imread("image.png")
# cv.imshow("image", img)
# cv.waitKey(0)

# we can also create a blank image 
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow("Blank", blank)
cv.waitKey(0)

# coloring the blank image 
# blank[y1:y2, x1:x2] = [B, G, R]
blank[:] = 0,255,0
cv.imshow("Green", blank)
cv.waitKey(0)

# drawing a rectangle 
# cv.rectangle(image, start_point, end_point, color, thickness)
# thickness=-1 or cv.FILLED will fill the shape
cv.rectangle(blank, (0,0), (250,250), (255,0,0), thickness=2)
cv.imshow("Rectangle", blank)
cv.waitKey(0)

# drawing a circle 
# cv.circle(image, center, radius, color, thickness)
cv.circle(blank, (250,250), 40, (0,0,255), thickness=3)
cv.imshow("Circle", blank)
cv.waitKey(0)

# drawing a line 
# cv.line(image, start_point, end_point, color, thickness)
cv.line(blank, (0,0), (500,500), (255,255,255), thickness=2)
cv.imshow("Line", blank)
cv.waitKey(0)

# adding text 
# cv.putText(image, text, org, font, fontScale, color, thickness)
# what is org? -> bottom-left corner of the text string in the image starting point
cv.putText(blank, "Hello World", (250,250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
cv.imshow("Text", blank)
cv.waitKey(0)

cv.destroyAllWindows()
