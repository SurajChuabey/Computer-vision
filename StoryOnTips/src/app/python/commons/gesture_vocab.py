import cv2
from src.app.python.MediaPipeModule.HandTracker.handtracker import HandTracker

class GestureVocabulary:
    """Class to handle gesture vocabulary using MediaPipe Hand Tracking."""

    def __init__(self):
        self.handT = HandTracker()
        self.tipPoints = [4, 8, 12, 16, 20]
        self.middlePoints = [2, 6, 10, 14, 18]
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
        other_finger_closed = True
        
        if len(positions)>=21:
            for bp,tp in zip(self.basePoints,self.tipPoints):
                if draw:
                    cv2.circle(image,(positions[bp][1],positions[bp][2]),5,(0,255,0),cv2.FILLED) 
                    cv2.circle(image,(positions[tp][1],positions[tp][2]),5,(255,0,0),cv2.FILLED)       
                    
                if tp == 4:
                    if abs(positions[tp][1] - positions[bp][1])>20:
                        other_finger_closed  = False
                elif tp == 8:
                    if positions[tp][2] < positions[bp][2]:
                        is_index_finger_up = True
                else:
                    if positions[tp][2] < positions[bp][2]:
                        other_finger_closed = False        
                        
        return is_index_finger_up and other_finger_closed,image
        
    def MiddleFingerUp_Dog(self,image,draw=False):
        """Detect middle finger up gesture to recognise dog in the image."""
        image = self.handT.findHands(image=image,draw=draw)
        positions,image =self.handT.findPosition(image=image,draw=draw)

        is_middle_finger_up = False
        other_finger_closed = True

        if len(positions)>=21:
            for bp,tp,mp in zip(self.basePoints,self.tipPoints,self.middlePoints):
                if draw:
                    cv2.circle(image,(positions[bp][1],positions[bp][2]),5,(0.255,0),cv2.FILLED)
                    cv2.circle(image,(positions[bp][1],positions[bp][2]),5,(0.255,0),cv2.FILLED)

                if tp == 4:
                    if positions[tp]:
                        if abs(positions[tp][1] - positions[bp][1])>40:
                            other_finger_closed  = False
                elif tp == 12:
                    if positions[tp][2] < positions[bp][2]:
                        is_middle_finger_up = True
                else:
                    if positions[tp][2] < positions[mp][2]:
                        other_finger_closed = False
        
        return is_middle_finger_up and other_finger_closed,image
    
    def IndexMiddleFingerUp_friend(self,image,draw=False):
        """Detect middle finger and index finger up gesture to recognise friend in the image."""
        image = self.handT.findHands(image=image,draw=draw)
        positions,image = self.handT.findPosition(image=image,draw=draw)

        is_index_finger_up = False
        is_middle_finger_up = False
        other_finger_closed = True

        if len(positions)>=21:
            for bp,tp,mp in zip(self.basePoints,self.tipPoints,self.middlePoints):
                if draw:
                    cv2.circle(image,(positions[bp][1],positions[bp][2]),5,(0.255,0),cv2.FILLED)
                    cv2.circle(image,(positions[bp][1],positions[bp][2]),5,(0.255,0),cv2.FILLED)
                if tp == 4:
                    if positions[tp]:
                        if abs(positions[mp][1]+1 - positions[5][1])>15:
                            other_finger_closed  = False
                elif tp == 12:
                    if positions[tp][2] < positions[mp][2]:
                        is_middle_finger_up = True
                elif tp == 8:
                    if positions[tp][2] < positions[mp][2]:
                        is_index_finger_up = True
                else:
                    if positions[tp][2] < positions[mp][2]:
                        other_finger_closed = False 

        return is_index_finger_up and is_middle_finger_up and other_finger_closed,image
    
    def OkSign(self,image,draw=False):
        """Detect ok sign gesture to recognise ok in the image."""
        image = self.handT.findHands(image=image,draw=draw)
        positions,image = self.handT.findPosition(image=image,draw=draw)

        is_pinched = False
        other_finger_up = True

        if len(positions)>=21:
            dx = positions[4][1] - positions[8][1]
            dy = positions[4][2] - positions[8][2]
            dist_sq = dx * dx + dy * dy
            threshold_sq = 1600 
            if dist_sq < threshold_sq:
                is_pinched = True

            for tp,mp in zip(self.tipPoints[2:],self.middlePoints[2:]):
                cv2.circle(image, (positions[tp][1], positions[tp][2]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (positions[mp][1], positions[mp][2]), 5, (0, 255, 0), cv2.FILLED)
                if positions[tp][2]> positions[mp][2]:
                    other_finger_up = False

        return is_pinched and other_finger_up,image
    
    def ThumbsUp_Good(self,image,draw=False):
        """Detect Thumb up gesture to recognise good in the image."""
        mage = self.handT.findHands(image=image,draw=draw)
        positions,image = self.handT.findPosition(image=image,draw=draw)

        is_thumb_up = False
        other_finger_closed = True
        finger_closed_thresh_sq = 1400

        def _dist_sq(a, b):
            dx = positions[a][1] - positions[b][1]
            dy = positions[a][2] - positions[b][2]
            return dx * dx + dy * dy
        
        if len(positions)>=21:
            for tp,mp in zip(self.tipPoints,self.middlePoints):
                if tp == 4:
                    if positions[tp][2]< positions[3][2]:
                        cv2.putText(image, f"{positions[tp][1],positions[tp][2]}", (positions[tp][1], positions[tp][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(image, f"{positions[mp][1],positions[mp][2]}", (positions[mp][1], positions[mp][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        is_thumb_up = True
                else:
                    if _dist_sq(tp, mp) > finger_closed_thresh_sq:
                        other_finger_closed = False
                        
        return is_thumb_up and other_finger_closed,image