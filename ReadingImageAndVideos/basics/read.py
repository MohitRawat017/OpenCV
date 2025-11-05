import cv2 as cv 

# img = cv.imread("image.png")
# # cv.imread() reads an image from the specified file and returns it as a NumPy array.
# cv.imshow("Image", img) # type: ignore
# # cv.imshow() displays an image in a window.
# cv.waitKey(0)
# # cv.waitKey(0) waits indefinitely for a key event before closing the window.

# # the problem with this is that large images may not fit on the screen
# # so we need to resize them before displaying


# WORKING WITH VIDEOS

capture = cv.VideoCapture("video.mp4")
# cv.VideoCapture() opens a video file or a capturing device for video capturing.

while True:
    isTrue, frame = capture.read()
    # capture.read() reads the next frame from the video file or capturing device.
    
    if not isTrue:
        break

    cv.imshow("Video", frame) # type: ignore
    # cv.imshow() displays the current frame in a window.

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    # cv.waitKey(20) waits for 20 milliseconds for a key event. If 'd' is pressed, the loop breaks.

# the above is basically reading images frame by frame from the video 
capture.release()
# capture.release() releases the video capturing object.
cv.destroyAllWindows()
