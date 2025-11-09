import cv2
import numpy as np
from PIL import Image

# --------------------------
# Utility: compute HSV limits
# --------------------------
def get_limits(bgr_color, hue_margin=10, min_sat=100, min_val=100):
    """
    Convert a BGR color to HSV and return lower+upper HSV limits,
    handling hue wrap-around (important for red).
    """
    # create a 1x1 image containing the BGR color, convert to HSV
    color = np.uint8([[bgr_color]])                     # shape: (1,1,3)
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]  # take the single pixel -> [H, S, V]

    hue = int(hsv_color[0])

    # Build preliminary limits around the hue
    lower_h = hue - hue_margin
    upper_h = hue + hue_margin

    # Clamp hue to [0, 179] and handle wrap-around by splitting ranges if needed.
    # For simplicity we clamp here; for perfect red handling you'd create two masks.
    lower_h = max(lower_h, 0)
    upper_h = min(upper_h, 179)

    lower_limit = np.array([lower_h, min_sat, min_val], dtype=np.uint8)
    upper_limit = np.array([upper_h, 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit

# --------------------------
# Main: video capture + detection
# --------------------------
def main():
    # Define BGR colors to test (OpenCV uses BGR by default)
    yellow_bgr = [0, 255, 255]
    red_bgr    = [0, 0, 255]
    blue_bgr   = [255, 0, 0]
    green_bgr  = [0, 255, 0]

    # Choose which color to detect (change this to test others)
    target_bgr = green_bgr

    # Open default webcam (0). If you have multiple cameras, change the index.
    cap = cv2.VideoCapture(0)

    # Kernel for morphological operations (noise removal)
    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        # If frame wasn't grabbed correctly, break the loop (prevents errors)
        if not ret:
            break

        # Convert BGR frame to HSV for better color segmentation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get HSV lower and upper bounds for the chosen BGR color
        lower_limit, upper_limit = get_limits(target_bgr)

        # Create a binary mask where white pixels are within the color range
        mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

        # --- Noise reduction to reduce flicker/glitch ---
        # Small median blur removes salt-and-pepper noise
        mask = cv2.medianBlur(mask, 5)
        # Morphological open: erode then dilate to remove small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Optional: dilate a bit to make detected area more continuous
        mask = cv2.dilate(mask, kernel, iterations=1)
        # ------------------------------------------------

        # Convert mask (numpy array) to PIL image so we can use getbbox()
        mask_pil = Image.fromarray(mask)
        bound_box = mask_pil.getbbox()  # returns (left, upper, right, lower) or None

        # If a bounding box exists, draw it on the original frame
        if bound_box:
            x1, y1, x2, y2 = bound_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Optionally show the mask too (handy for debugging)
        cv2.imshow('Mask', mask)

        # Show the original frame with rectangle
        cv2.imshow('Frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
