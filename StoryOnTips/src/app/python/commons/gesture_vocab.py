import cv2
from src.app.python.MediaPipeModule.HandTracker.handtracker import HandTracker

class GestureVocabulary:
    """Class to handle gesture vocabulary using MediaPipe Hand Tracking."""

    def __init__(self):
        self.handT = HandTracker()
        self.tipPoints = [4, 8, 12, 16, 20]
        self.basePoints = [1, 5, 9, 13, 17]

    def OpenPalm_Tree(self, image, draw=False):
        """Detect Open Palm Tree gesture in the image."""
        image = self.handT.findHands(image, draw=draw)
        positions, image = self.handT.findPosition(image, draw=draw)

        is_Palm_tree = False 

        if len(positions) >= 21:
            all_fingers_extended = True

            for bp, tp in zip(self.basePoints, self.tipPoints):
                if draw:
                    cv2.circle(image, (positions[bp][1], positions[bp][2]), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(image, (positions[tp][1], positions[tp][2]), 5, (255, 0, 0), cv2.FILLED)

                # For fingers except thumb: check Y
                if tp != 4:
                    if positions[tp][2] >= positions[bp][2]:
                        all_fingers_extended = False
                else:
                    if abs(positions[tp][1] - positions[bp][1]) < 20:
                        all_fingers_extended = False

            is_Palm_tree = all_fingers_extended

        return is_Palm_tree, image

    def ClosedFist_earth(self, image , draw = False):
        """Detect Closed Fist Earth gesture in the image."""
        image = self.handT.findHands(image=image, draw=draw)
        postions,image = self.handT.findPosition(image=image, draw=draw)

        is_closed_fist = False

        if len(postions)>= 21:
            all_fingere_closed = True
            for bp,tp in zip(self.basePoints, self.tipPoints):
                if draw:
                    cv2.circle(image, (postions[bp][1], postions[bp][2]), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(image, (postions[tp][1], postions[tp][2]), 5, (255, 0, 0), cv2.FILLED)

                if tp!=4:
                    if postions[tp][2] <= postions[bp][2]:
                        all_fingere_closed = False
                else:
                    if abs(postions[tp][1] - postions[bp][1]) > 40:
                        all_fingere_closed = False
                        
            is_closed_fist = all_fingere_closed

        return is_closed_fist, image
        
        
    def IndexFingerUp_run(self, image ,draw=False):
        """Detect index finger up gesture to recognise run in the image."""
        image = self.handT.findHands(image=image , draw=draw)
        positions,image = self.handT.findPosition(image=image , draw=draw)
        
        is_index_finger_up = False
        
        if len(positions)>=21:
            other_finger_closed = True
            for bp,tp in zip(self.basePoints,self,self.tipPoints):
                if draw:
                    cv2.circle(image,(positions[bp][1],positions[bp][2]),5,(0,255,0),cv2.FILLED) 
                    cv2.circle(image,(positions[tp][1],positions[tp][2]),5,(255,0,0),cv2.FILLED)       
                    
                
                if tp!=8:
                    if positions[tp][2]< positions[bp][2]:
                        other_finger_closed = False
                else:
                    if positions[tp][2] < positions[bp][2]:
                        is_index_finger_up = True
                        
        return is_index_finger_up and other_finger_closed
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
