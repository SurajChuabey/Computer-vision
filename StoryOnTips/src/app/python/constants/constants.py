import os

class Constants:
    """A class to hold constants for the HandTracker module."""
    
    BASE_DIR = os.path.abspath(os.curdir)

    # MediaPipe Hands model complexity
    MODEL_COMPLEXITY = 1  # 0: Fast, 1: Full, 2: Heavy
    
    # Detection confidence threshold
    DETECTION_CONFIDENCE = 0.5
    
    # Tracking confidence threshold
    TRACKING_CONFIDENCE = 0.5
    
    # Drawing specifications for landmarks and connections
    DRAW_SPEC = {
        'landmark_color': (255, 0, 255),
        'connection_color': (0, 255, 0),
        'landmark_radius': 5,
        'connection_thickness': 2
    }