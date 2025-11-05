import cv2 as cv 

# create a rectangle and a circle
rectangle = cv.rectangle(
    img = cv.imread("../image.png"),
    pt1 = (200, 200),
    pt2 = (400, 400),
    color=(255, 255, 255),
    thickness = -1
)

circle = cv.circle(
    img = cv.imread("../image.png"),
    center = (300, 300),
    color=(255, 255, 255),
    radius = 100,
    thickness = -1
)

# Bitwise AND --> intersection
bitwise_and = cv.bitwise_and(src1 = rectangle, src2 = circle)
cv.imshow("Bitwise AND", bitwise_and)

# Bitwise OR --> union
bitwise_or = cv.bitwise_or(src1 = rectangle, src2 = circle)
cv.imshow("Bitwise OR", bitwise_or)

# Bitwise XOR --> exclusive or
bitwise_xor = cv.bitwise_xor(src1 = rectangle, src2 = circle)
cv.imshow("Bitwise XOR", bitwise_xor)

# Bitwise NOT --> inverse
bitwise_not_rectangle = cv.bitwise_not(src = rectangle)
cv.imshow("Bitwise NOT Rectangle", bitwise_not_rectangle)

bitwise_not_circle = cv.bitwise_not(src = circle)
cv.imshow("Bitwise NOT Circle", bitwise_not_circle)

cv.waitKey(0)
cv.destroyAllWindows()