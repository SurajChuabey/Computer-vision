from src.app.python.commons.gesture_vocab import GestureVocabulary
import cv2

def main():
    """Main function to test the Open Palm Tree gesture detection."""
    vocab = GestureVocabulary()
    video_path= 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Unable to open video path {video_path}")

    while cap.isOpened():

        ret,frame = cap.read()
        if not ret:
            print("Unable to read frame")

        # is_palm_opened,annotated_frame = vocab.OpenPalm_Tree(frame,draw=False)
        # is_fist_closed, annotated_frame = vocab.ClosedFist_earth(frame, draw=True)
        # is_index_finger_up, annotated_frame = vocab.IndexFingerUp_run(frame, draw=True)
        # is_middle_finger_up, annotated_frame = vocab.MiddleFingerUp_Dog(frame, draw=False)


        # if is_palm_opened:
        #     cv2.putText(annotated_frame, "Palm Opened:Recognise Tree", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(annotated_frame, "Palm Closed:Unable to recognise", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # if is_fist_closed:
        #     cv2.putText(annotated_frame, "Fist is closed :recognise earth", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(annotated_frame, "Fist is not closed:Unable to recognise", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # if is_index_finger_up:
        #     cv2.putText(annotated_frame, "Index Finger Up: Recognised Run", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(annotated_frame, "Index Finger Not Up: Unable to recognise", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # if is_middle_finger_up:
        #     cv2.putText(annotated_frame, "Middle Finger Up: Recognised Dog", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(annotated_frame, "Middle Finger Not Up: Unable to recognise", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # cv2.imshow("Frame", annotated_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()