from predict_cam import predict
from Gazetracker import gaze_tracker
import cv2

if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    while True:
        status, frame = webcam.read()
        if not status:
            break
        frame = cv2.resize(frame, (1280, 720))
        predict1,predict2=predict(frame)
        if predict1==('safe_driving'):
            gazeout = gaze_tracker(frame)
            if (gazeout == 'looking_left') or (gazeout=="Looking right") \
                    or (gazeout=="Looking up") or (gazeout=="Looking down"):
                cv2.putText(frame, 'look downside', (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
            else:
                cv2.putText(frame, str(predict1) + str(predict2), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0),
                            1)
        else:
            cv2.putText(frame, str(predict1) + str(predict2), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)
        cv2.imshow("img", frame)
        cv2.waitKey(1)