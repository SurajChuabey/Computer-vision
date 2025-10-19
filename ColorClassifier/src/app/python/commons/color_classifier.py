import cv2
import os
import numpy as np
from src.app.python.constants.constants import Constants
from src.app.python.commons.config_reader import cfg

class ColorClassifier:
    """This class classifies regions based on the colors present in a given frame"""

    def __init__(self):
        self.contour_area_threshold = float(os.environ.get(Constants.CONTOUR_AREA_THRESHOLD,cfg.get_env_config(Constants.CONTOUR_AREA_THRESHOLD)))
        self.threshold_value = int(os.environ.get(Constants.BINARY_THRESHOLD_VALUE,cfg.get_env_config(Constants.BINARY_THRESHOLD_VALUE)))

    def classify_color_hsv(self, h, s, v):
        """Classifies the color based on HSV values."""
        if v < Constants.BLACK_THRESHOLD:
            return Constants.COLOR_BLACK
        elif s < Constants.WHITE_THRESHOLD_SATURATION and v > Constants.WHITE_THRESHOLD_VALUE:
            return Constants.COLOR_WHITE
        elif s < Constants.GRAY_THRESHOLD:
            return Constants.COLOR_GRAY
        elif h < Constants.RED_HUE_LOWER or h > Constants.RED_HUE_UPPER:
            return Constants.COLOR_RED
        elif Constants.ORANGE_HUE_LOWER < h <= Constants.ORANGE_HUE_UPPER:
            return Constants.COLOR_ORANGE
        elif Constants.YELLOW_HUE_LOWER < h <= Constants.YELLOW_HUE_UPPER:
            return Constants.COLOR_YELLOW
        elif Constants.GREEN_HUE_LOWER < h <= Constants.GREEN_HUE_UPPER:
            return Constants.COLOR_GREEN
        elif Constants.BLUE_HUE_LOWER < h <= Constants.BLUE_HUE_UPPER:
            return Constants.COLOR_BLUE
        elif Constants.PURPLE_HUE_LOWER < h <= Constants.PURPLE_HUE_UPPER:
            return Constants.COLOR_PURPLE
        else:
            return Constants.COLOR_UNKNOWN

    def get_regions(self, frame: np.ndarray):
        """Identifies regions in the frame and classifies their colors."""
        marked_regions = {}
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey_frame, self.threshold_value, Constants.MAX_THRESHOLD_VALUE, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > self.contour_area_threshold]

        for i, contour in enumerate(filtered_contours):
            mask = np.zeros_like(grey_frame)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Compute the mean HSV values within the mask
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv_frame, mask=mask)[:3]
            label = self.classify_color_hsv(*mean_hsv)

            marked_regions[f"region_{i}"] = (contour, label)

        return marked_regions
    
    def draw_regions(self, frame: np.ndarray, regions: dict):
        """Draws the contours and labels on the frame."""
        for region_name, (contour, label) in regions.items():
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), Constants.TWO)

            # Find centroid
            M = cv2.moments(contour)
            if M["m00"] != Constants.ZERO:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = contour[Constants.ZERO][Constants.ZERO]

            cv2.putText(frame, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), Constants.TWO)

        return frame
    
    def classify_color(self, frame: np.ndarray):
        """Classifies the colors in the given frame."""
        regions = self.get_regions(frame)
        annotated_frame = self.draw_regions(frame, regions)
        return annotated_frame    