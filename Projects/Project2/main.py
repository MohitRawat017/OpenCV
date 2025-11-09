import cv2 as cv
import argparse
import mediapipe as mp


def process_image(img, face_detection, resize_scale=1.0):

    if resize_scale != 1.0:
        img = cv.resize(img, None, fx=resize_scale, fy=resize_scale, interpolation=cv.INTER_AREA)

    height, width, channels = img.shape
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    output = face_detection.process(img_rgb)

    if output.detections is not None:
        for detection in output.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Convert relative coordinates (0–1) to absolute pixel coordinates
            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            w  = int(bbox.width * width)
            h  = int(bbox.height * height)

            # Clip coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x1 + w)
            y2 = min(height, y1 + h)

            # Skip invalid or empty regions
            if x2 <= x1 or y2 <= y1:
                continue

            # Apply blur safely
            face_roi = img[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            img[y1:y2, x1:x2] = cv.GaussianBlur(face_roi, ksize=(41, 41), sigmaX=30)

    return img


parser = argparse.ArgumentParser()
# it is used to parse command-line arguments for the script.
parser.add_argument('--mode', default='webcam')
parser.add_argument('--filePath', default='0')
parser.add_argument('--resize_scale', type=float, default=1.0)
parser.add_argument('--distance', type=int, default=0)
parser.add_argument('--confidence', type=float, default=0.5)
args = parser.parse_args()

# python main.py --mode video --filePath video.mp4 # for video and similarily for image , webcam is default


mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=args.distance, min_detection_confidence=args.confidence) as face_detection:
    if args.mode in ['image']:
        img = cv.imread(args.filePath)
        

        img = process_image(img, face_detection, resize_scale=args.resize_scale)
        cv.imwrite('output.png', img)

    # video 
    elif args.mode in ['video']:

        
        cap = cv.VideoCapture(args.filePath)
        fps = cap.get(cv.CAP_PROP_FPS)
        # Read first frame to get dimensions
        ret, frame = cap.read()

        if fps == 0:
            fps = 30 # default fps value

        height, width, channels = frame.shape
        output_video = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while ret:
            
            if not ret:
                break

            frame = process_image(frame, face_detection=face_detection, resize_scale=args.resize_scale)

            output_video.write(frame)
            
            cv.imshow('Video', frame)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()

        
        cap.release()
        output_video.release()
        cv.destroyAllWindows()
    
    elif args.mode in ['webcam']:
        cap = cv.VideoCapture(0)

        ret, frame = cap.read()
        height, width, channels = frame.shape
        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_image(frame, face_detection=face_detection)

            cv.imshow('Webcam', frame)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()



