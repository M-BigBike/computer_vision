# Re-run the image loading and processing with the updated technique using HSV and median filter

# Homework 2.4
import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_coins(image_path):
    # Load the image
    image = cv2.imread('C:/Users/Ez-Studio/computer_vision_660632034/dataset/COIN/CoinCounting/coin3.jpg')
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and blue in HSV
    # Note: These ranges can be fine-tuned
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

    # Find contours for the yellow and blue masks
    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Draw contours and print color name on the original image
    for cnt in contours_yellow:
        cv2.drawContours(image, [cnt], 0, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(image, 'Yellow', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for cnt in contours_blue:
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(image, 'Blue', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Count the number of contours found for each color
    yellow_count = len(contours_yellow)
    blue_count = len(contours_blue)

    return yellow_count, blue_count, image, mask_yellow, mask_blue


def detect_coins_with_morphology(image_path):
    # Load the image
    image = cv2.imread('C:/Users/Ez-Studio/computer_vision_660632034/dataset/COIN/CoinCounting/coin3.jpg')
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and blue in HSV
    # Note: These ranges can be fine-tuned
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    lower_blue = np.array([100, 150, 0], dtype="uint8")
    upper_blue = np.array([140, 255, 255], dtype="uint8")

    # Create masks for yellow and blue colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Define the structuring element for morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply opening to remove noise
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    # Apply closing to close small holes inside the foreground
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    # Find contours for the yellow and blue masks
    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Filter out very small contours that are likely not coins
    contours_blue = [cnt for cnt in contours_blue if cv2.contourArea(cnt) > 100]

    # Draw contours and print color name on the original image
    for cnt in contours_yellow:
        cv2.drawContours(image, [cnt], 0, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(image, 'Yellow', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for cnt in contours_blue:
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(image, 'Blue', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Count the number of contours found for each color
    yellow_count = len(contours_yellow)
    blue_count = len(contours_blue)

    return yellow_count, blue_count, image, mask_yellow, mask_blue


# Instead of image_paths[0], use image_path
results_morphology = detect_coins_with_morphology(image_path)

# Show the original image with detected contours and masks
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(results_morphology[2], cv2.COLOR_BGR2RGB))
plt.title('Detected Coins')

plt.subplot(1, 3, 2)
plt.imshow(results_morphology[3], cmap='gray')
plt.title('Yellow Mask')

plt.subplot(1, 3, 3)
plt.imshow(results_morphology[4], cmap='gray')
plt.title('Blue Mask')

plt.tight_layout()
plt.show()

# Return the count of yellow and blue coins
# results_morphology[0], results_morphology[1]

all_results = []

for image_path in image_paths:
    yellow_count, blue_count, _ = detect_coins(image_path)
    all_results.append({
        'file_name': Path(image_path).name,
        'yellow_count': yellow_count,
        'blue_count': blue_count
    })

# Display the results
all_results

