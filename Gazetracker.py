# coding=UTF-8
"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import time

# webcam = cv2.VideoCapture('DJI_0068.MP4')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# frame_count = webcam.get(cv2.CAP_PROP_FPS)
# frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
# out = cv2.VideoWriter('out.mp4', fourcc, frame_count, (frame_width, frame_height))

def gaze_tracker(img):
    frame = img
    gaze = GazeTracking()
    t1 = time.perf_counter()
    # We get a new frame from the webcam

    # We send this frame to GazeTracking to analyze it
    # frame = cv2.resize(frame,(720, int(720*frame_height/frame_width)))
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_up():
        text = "Looking up"
    elif gaze.is_down():
        text = "Looking down"
    elif gaze.is_center():
        text = "Looking center"
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    horizontal_ratio = gaze.horizontal_ratio()
    vertical_ratio = gaze.vertical_ratio()
    blink_ratio = gaze.blink_ratio()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Horizontal_ratio: " + str(horizontal_ratio), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Vertical_ratio: " + str(vertical_ratio), (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Blink_ratio: " + str(blink_ratio), (90, 270), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    if left_pupil !=  None and right_pupil != None and gaze.is_blinking() == False:
        left_pupil_out = (left_pupil[0] + int(300 * (horizontal_ratio - 0.5)), left_pupil[1] + int(100 * (vertical_ratio - 0.5)))
        cv2.arrowedLine(frame, left_pupil, left_pupil_out, (0, 0, 255), 2, 0, 0, 0.2)
        right_pupil_out = (right_pupil[0] + int(300 * (horizontal_ratio - 0.5)), right_pupil[1] + int(100 * (vertical_ratio - 0.5)))
        cv2.arrowedLine(frame, right_pupil, right_pupil_out, (0, 0, 255), 2, 0, 0, 0.2)
    # out.write(frame)
    cv2.imshow("Demo", frame)
    cv2.waitKey(1)
    return text