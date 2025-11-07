import cv2 as cv
import mediapipe as mp
# MediaPipe face detection module is a module for detecting faces in images and videos.
# read image 
img = cv.imread('image.png')

height , width, channels = img.shape


# detect faces
mp_face_detection = mp.solutions.face_detection
# Initialize the face detection model

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
# model_selection: 0 for short-range (2 meters), 1 for full-range (5 meters)
# min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
    
    # Convert the BGR image to RGB before processing.
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    output = face_detection.process(img_rgb)
    # print(output.detections) # we can see the detected faces information

    # extract bounding box info
    if output.detections is not None :
        for detection in output.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box 

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            # we have got the bounding box values in relative format
            x1, y1, w, h = int(x1 * width), int(y1 * height), int(w * width), int(h * height)
            # Because the bounding box values are in relative format (0 to 1), we need to convert them to absolute pixel values by multiplying with the image dimensions.

            # cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

            # Blur the detected faces
            img[y1:y1+h, x1:x1+w] = cv.blur(img[y1:y1+h, x1:x1+w], ksize=(30,30))


# save the output image 
cv.imwrite('output.png', img)