import os
from src.app.python.main import main
from src.app.python.constants.constants import Constants

VIDEO_PATH = os.environ.get(Constants.VIDEO_PATH, None)
IMAGE_PATH = os.environ.get(Constants.IMAGE_PATH, None)

if __name__ == "__main__":
    main(video_path=VIDEO_PATH, image_path=IMAGE_PATH)
    # This will start the application and run the main function defined in main.py