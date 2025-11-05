import cv2 as cv
from matplotlib.pyplot import gray 


# ## TO DETECT FACES IN A STATIC IMAGE

# img = cv.imread('../image.png')
# cv.imshow('Image', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Image', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
# # scaleFactor: How much the image size is reduced at each image scale.
# # minNeighbors: the less the more detections but also more false positives
# print(f'Number of faces found = {len(faces_rect)}')

# for (x, y, w, h) in faces_rect:
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
#     # (x, y): Top-left corner of the rectangle
#     # (x + w, y + h): Bottom-right corner of the rectangle
# cv.imshow('Detected Faces', img)

## TO DETECT FACES IN A VIDEO
video = cv.VideoCapture('../video.mp4')
while True:
    isTrue, frame = video.read()
    if not isTrue:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.imshow('Video Face Detection', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
video.release()
cv.destroyAllWindows()