import numpy as np
import cv2

def get_limits(bgr_color):
    """
    Convert a BGR color into its corresponding lower and upper HSV limits
    for color detection in OpenCV.
    """

    # ----------------------------------------------
    # Step 1: Wrap the color into a tiny NumPy array
    # ----------------------------------------------
    # What: create a 1x1 image (1 pixel) with our target color.
    # How: cv2.cvtColor expects an image-like array, so we simulate that.
    # Why: cv2.cvtColor works only on images, not single [B, G, R] lists.
    # Friend: “We trick OpenCV by pretending our color is a 1-pixel image.”
    color = np.uint8([[bgr_color]])  # shape = (1, 1, 3)

    # ----------------------------------------------
    # Step 2: Convert BGR → HSV color space
    # ----------------------------------------------
    # What: Convert that 1-pixel BGR image to HSV.
    # How: cv2.cvtColor changes color representation; [0][0] extracts the single HSV pixel.
    # Why: HSV separates Hue (color type), Saturation (intensity), and Value (brightness),
    #      which makes color detection easier and lighting-independent.
    # Friend: “We turn it into HSV so we can find its exact hue number.”
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]

    # ----------------------------------------------
    # Step 3: Extract the hue (main color angle)
    # ----------------------------------------------
    # What: Get the Hue value (0–179 in OpenCV).
    # How: hsv_color is [H, S, V], so hsv_color[0] is hue.
    # Why: Hue is what defines the actual color (red, yellow, blue…)
    # Friend: “Hue is the number that tells OpenCV what color family it is.”
    hue = hsv_color[0]

    # ----------------------------------------------
    # Step 4: Build lower and upper HSV limits
    # ----------------------------------------------
    # What: Define a range of ±10 around the hue.
    # How: Subtract 10 for lower bound, add 10 for upper bound.
    # Why: Colors aren’t exact — light and reflections change them slightly,
    #      so we allow a small margin.
    # Friend: “We give a 10-degree window around that color so it still works
    #          if lighting or shade changes.”
    lower_limit = np.array([hue - 10, 100, 100])
    upper_limit = np.array([hue + 10, 255, 255])

    # ----------------------------------------------
    # Step 5: Handle special case — red hue wrap-around
    # ----------------------------------------------
    # What: Red in HSV lies at both ends (near 0 and 179).
    # How: If hue - 10 < 0 or hue + 10 > 179, we clamp values to valid range.
    # Why: Hue wraps around (it’s circular). Without this, OpenCV might get
    #      negative hue values or >179, which don’t exist.
    # Friend: “Imagine a color wheel — red is right at the edge, so we make sure
    #          the range doesn’t go off the circle.”
    if hue - 10 < 0:
        lower_limit[0] = 0
    if hue + 10 > 179:
        upper_limit[0] = 179

    # ----------------------------------------------
    # Step 6: Convert limits back to uint8
    # ----------------------------------------------
    # What: Ensure they’re in correct format for OpenCV functions.
    # How: Cast both limits to np.uint8 (0–255 range).
    # Why: cv2.inRange() requires uint8 arrays.
    # Friend: “Just making sure OpenCV can understand these values.”
    return lower_limit.astype(np.uint8), upper_limit.astype(np.uint8)
