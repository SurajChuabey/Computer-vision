import os
import cv2
import uuid
from src.app.python.commons.color_classifier import ColorClassifier
from src.app.python.commons.config_reader import cfg
from src.app.python.constants.constants import Constants
classifier = ColorClassifier()


def main(image_path: str = None, video_path: str = None):
    """The main function to run the color classifier. on the given frame and video."""
    env_mode = os.getenv(Constants.ENV_MODE,cfg.get_env_config(Constants.ENV_MODE))

    if env_mode == Constants.ENV_MODE_FRAME:
        frame = cv2.imread(image_path) if image_path else None
        if frame is None:
            raise ValueError("Frame cannot be None in ENV_MODE_FRAME")
        annotated_frame = classifier.classify_color(frame=frame)
        cv2.imwrite(os.path.join(Constants.SAVED_FRAME_NAME,f"{uuid.uuid4}.{Constants.FRAME_EXTENSION_JPG}"), annotated_frame)
        return annotated_frame
    
    elif env_mode == Constants.ENV_MODE_VIDEO:
        if video_path is None:
            raise ValueError("Video cannot be None in ENV_MODE_VIDEO")
        if not os.path.exists(Constants.ARTIFACTS_DIR):
            os.makedirs(Constants.ARTIFACTS_DIR)
        out = cv2.VideoWriter(os.path.join(Constants.ARTIFACTS_DIR,Constants.SAVED_VIDEO_NAME),cv2.VideoWriter_fourcc(*Constants.CODEC),Constants.FPS,(Constants.FRAME_WIDTH,Constants.FRAME_HEIGHT))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            annotated_frame = classifier.classify_color(frame=frame)
            cv2.imshow("Color Classifier", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            resized_frame = cv2.resize(annotated_frame, (Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT))
            out.write(resized_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        raise ValueError(f"Invalid ENV_MODE: {env_mode}. Expected 'frame' or 'video'.")
    return None