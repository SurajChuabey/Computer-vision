import cv2
import numpy as np
import mediapipe as mp

class HandTracker:
    """A class to track hands using MediaPipe Hands module."""

    def __init__(self):
        """Initialize the HandTracker with MediaPipe Hands settings."""
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)
        self.results = None

    def findHands(self,image, draw = False):
        """Detect hands in the image and return the image with landmarks if draw is True."""
        imageRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS, self.drawSpec, self.drawSpec)
                return image
            else:
                return image
            
        else:
            if draw:
                cv2.putText(image, "No Hands Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return image

    def findPosition(self, image, handno = 0 ,draw = False):
        """Find the position of landmarks in the image."""
        positions = []
        if self.results.multi_hand_landmarks:
            if handno < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handno]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    positions.append([id, cx, cy])
                    if draw:
                        cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return positions,image