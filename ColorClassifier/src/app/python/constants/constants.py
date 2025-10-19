import os

class Constants:
    # Base directory for the application
    BASE_DIR = os.path.abspath(os.curdir)
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
    ENV_MODE = 'ENV_MODE'
    ENV_MODE_FRAME = 'FRAME'
    ENV_MODE_VIDEO = 'VIDEO'
    SAVED_FRAME_NAME = 'saved_frame'
    SAVED_VIDEO_NAME = 'saved_video.mp4'
    IMAGE_PATH = "IMAGE_PATH"
    VIDEO_PATH = "VIDEO_PATH"
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
    FRAME_EXTENSION_JPG = '.jpg'
    FPS = 20
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Constants for color classification
    BLACK_THRESHOLD = 50
    WHITE_THRESHOLD_SATURATION = 50
    WHITE_THRESHOLD_VALUE = 200
    GRAY_THRESHOLD = 50
    RED_HUE_LOWER = 10
    RED_HUE_UPPER = 160
    ORANGE_HUE_LOWER = 10
    ORANGE_HUE_UPPER = 25
    YELLOW_HUE_LOWER = 25
    YELLOW_HUE_UPPER = 35
    GREEN_HUE_LOWER = 35
    GREEN_HUE_UPPER = 85
    BLUE_HUE_LOWER = 85
    BLUE_HUE_UPPER = 125
    PURPLE_HUE_LOWER = 125
    PURPLE_HUE_UPPER = 160
    MAX_THRESHOLD_VALUE = 255

    # Constants for region detection
    BINARY_THRESHOLD_VALUE = 'BINARY_THRESHOLD_VALUE'
    CONTOUR_AREA_THRESHOLD = 'CONTOUR_AREA_THRESHOLD'

    # Color names
    COLOR_BLACK = "Black"
    COLOR_WHITE = "White"
    COLOR_GRAY = "Gray"
    COLOR_RED = "Red"
    COLOR_ORANGE = "Orange"
    COLOR_YELLOW = "Yellow"
    COLOR_GREEN = "Green"
    COLOR_BLUE = "Blue"
    COLOR_PURPLE = "Purple"
    COLOR_UNKNOWN = "Unknown"


    # config parameters
    DEFAULT_ENVIRONMENT = 'DEFAULT'
    CONFIG_FILE_PATH = os.path.join(BASE_DIR, 'src/app/python/configurations/configuration.ini')
    UTF_8_ENCODING = 'utf-8'
    CODEC = 'mp4v'

    ZERO,ONE,TWO,THREE,FOUR,FIVE,SIX,SEVEN,EIGHT,NINE,TEN = range(0, 11)