import cv2
import numpy as np

image_path = "/home/suraj/Pictures/Wallpapers/7.jpg"

class ColorClassifier:
    """This class classifies regions based on the colors present in a given frame"""

    def __init__(self):
        pass

    def classify_color_hsv(self, h, s, v):
        if v < 50:
            return "Black"
        elif s < 50 and v > 200:
            return "White"
        elif s < 50:
            return "Gray"
        elif h < 10 or h > 160:
            return "Red"
        elif 10 < h <= 25:
            return "Orange"
        elif 25 < h <= 35:
            return "Yellow"
        elif 35 < h <= 85:
            return "Green"
        elif 85 < h <= 125:
            return "Blue"
        elif 125 < h <= 160:
            return "Purple"
        else:
            return "Unknown"

    def get_regions(self, frame: np.ndarray):
        marked_regions = {}
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey_frame, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 40]

        for i, contour in enumerate(filtered_contours):
            mask = np.zeros_like(grey_frame)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Compute the mean HSV values within the mask
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv_frame, mask=mask)[:3]
            label = self.classify_color_hsv(*mean_hsv)

            marked_regions[f"region_{i}"] = (contour, label)

        return marked_regions

# Load image
frame = cv2.imread(image_path)

# Process regions
cls = ColorClassifier()
regions = cls.get_regions(frame=frame)

# Draw labeled contours
for region_name, (contour, label) in regions.items():
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Find centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = contour[0][0]

    cv2.putText(frame, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Show result
cv2.imshow("Final Output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()