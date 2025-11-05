import cv2 as cv # type: ignore

img = cv.imread("Photos/image.png")

def rescaleFrame(frame, scale=0.75):
    # Works for images, videos and live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
# cv.resize() takes 3 arguments: image, dimensions, interpolation
# interpolation is optional and it decides the quality of the image after resizing

# for videos 
# method 1 :
# capture = cv.VideoCapture("video.mp4")
# while True:
#     isTrue, frame = capture.read()
#     frame_resized = rescaleFrame(frame)
#     cv.imshow("Video", frame_resized)
#     if cv.waitKey(20) & 0xFF == ord("d"):
#         break
# capture.release()
# cv.destroyAllWindows()


# method 2 : for live video
capture = cv.VideoCapture(0)
# 0 is for default camera, 1 for external camera
def changeRes(width, height):
    # live video
    capture.set(3, width)  # 3 is for width
    capture.set(4, height) # 4 is for height