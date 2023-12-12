# Re-run the image loading and processing with the updated technique using HSV and median filter

# Load the image
image = cv2.imread('C:/Users/Ez-Studio/computer_vision_660632034/dataset/COIN/CoinCounting/coin1.jpg')

# Check if the image is loaded properly
if image is not None and image.size > 0:
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color ranges for yellow and blue in HSV
    # These ranges are based on typical values for yellow and blue colors in the HSV space
    lower_yellow = np.array([22, 93, 0], dtype="uint8")
    upper_yellow = np.array([45, 255, 255], dtype="uint8")

    lower_blue = np.array([90, 60, 0], dtype="uint8")
    upper_blue = np.array([128, 255, 255], dtype="uint8")

    # Create masks for yellow and blue colors
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # Apply a median blur to the mask to filter out noise
    mask_yellow = cv2.medianBlur(mask_yellow, 15)
    mask_blue = cv2.medianBlur(mask_blue, 15)

    # Find contours for the yellow and blue masks
    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow = contours_yellow[0] if len(contours_yellow) == 2 else contours_yellow[1]
    contours_blue = contours_blue[0] if len(contours_blue) == 2 else contours_blue[1]

    # Count the number of contours found for each color
    yellow_count = len(contours_yellow)
    blue_count = len(contours_blue)
else:
    yellow_count, blue_count = 0, 0

(yellow_count, blue_count), (mask_yellow, mask_blue)
