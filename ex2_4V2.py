import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_coins_with_circle_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and blue in HSV
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    lower_blue = np.array([100, 150, 0], dtype="uint8")
    upper_blue = np.array([140, 255, 255], dtype="uint8")

    # Create masks for yellow and blue colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply median blur
    mask_yellow = cv2.medianBlur(mask_yellow, 15)
    mask_blue = cv2.medianBlur(mask_blue, 15)

    # Use Hough Circles to detect circles on the blue mask
    # cv2.HOUGH_GRADIENT is the circle detection method, dp is the inverse ratio of the accumulator resolution
    # to the image resolution, minDist is the minimum distance between the centers of the detected circles
    circles = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=60)

    # Initialize count for blue and yellow coins
    yellow_count = 0
    blue_count = 0

    # Draw the detected yellow circles and count them
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (255, 0, 0), 4)
            cv2.putText(image, 'Blue', (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            blue_count += 1

    # Find contours for the yellow mask to count yellow coins
    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours_yellow:
        cv2.drawContours(image, [cnt], 0, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(image, 'Yellow', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        yellow_count += 1

    return yellow_count, blue_count, image, mask_yellow, mask_blue

# Apply the detection with Hough Circle Transform to the uploaded image
coin4_path = 'C:/Users/Ez-Studio/computer_vision_660632034/dataset/COIN/CoinCounting/coin10.jpg'
results_circle_detection = detect_coins_with_circle_detection(coin4_path)

# Show the original image with detected contours and masks
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(results_circle_detection[2], cv2.COLOR_BGR2RGB))
plt.title('Detected Coins')

plt.subplot(1, 3, 2)
plt.imshow(results_circle_detection[3], cmap='gray')
plt.title('Yellow Mask')

plt.subplot(1, 3, 3)
plt.imshow(results_circle_detection[4], cmap='gray')
plt.title('Blue Mask')

plt.tight_layout()
plt.show()

# Return the count of yellow and blue coins
results_circle_detection[0], results_circle_detection[1]
