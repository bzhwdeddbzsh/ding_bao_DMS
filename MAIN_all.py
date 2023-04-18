from predict_cam import predict
from Gazetracker import gaze_tracker
import csv
import cv2
import os

def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            get_filelist(newDir, Filelist)
    return Filelist

if __name__ == '__main__':
    with open("out.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["drinking", "hair_and_makeup", "phonecall_left", "phonecall_right", "radio", "reach_backseat",
             "reach_side", "safe_drive","smoking","talking_to_passenger", "texting_left", "texting_right", 'img'])
    files=[]
    get_filelist('face_gZ',files)
    for file in files:
        row = [0,0,0,0,0,0,0,0,0,0,0,0,file]
        frame= cv2.imread(file)
        # frame = cv2.resize(frame, (1280, 720))
        predict1,predict2, outnum =predict(frame)
        if predict1==('safe_driving'):
            gazeout = gaze_tracker(frame)
            if (gazeout == 'Looking left') or (gazeout=="Looking right") \
                    or (gazeout=="Looking down"):
                cv2.putText(frame, 'look downside', (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
                out = 'look downside'
                outnum = 10
            else:
                cv2.putText(frame, str(predict1) + str(predict2), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0),
                            1)
                out = 'safe_driving'
        else:
            cv2.putText(frame, str(predict1) + str(predict2), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)
            out = predict1
        print(out)
        row[int(outnum)] = 1
        with open('out.csv', "a", newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(row)
        # cv2.imshow("img", frame)
        # cv2.waitKey(1)