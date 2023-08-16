import random
from gaze_tracking import GazeTracking
import time

import os
import json

import torch
import cv2
from torchvision import transforms
from PIL import Image

from model_cbam import efficientnet_b0 as create_model

def predict(img,model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    frame = img
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'model/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    out1= class_indict[str(predict_cla)]

    out2= predict[predict_cla].numpy()
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    frame = cv2.resize(frame, (1280, 720))
    cv2.putText(frame, print_res, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.imshow("img", frame)
    # cv2.waitKey(1)
    return out1,out2,str(predict_cla)

def setup_model(model_path='model/model-70_0.978_fix.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=12).to(device)
    # load model weights
    model_weight_path = model_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    return model

def gaze_tracker(img):

    frame = cv2.resize(img, (960, 540))
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
    # cv2.imshow("Demo", frame)
    # cv2.waitKey(1)
    return text



if __name__ == '__main__':
    webcam = cv2.VideoCapture('show.mp4')
    model= setup_model()
    #cv2保存视频
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('out.mp4', fourcc, 30, (1280, 720))

    while True:
        status, frame = webcam.read()
        if not status:
            break
        frame = cv2.resize(frame, (1280, 720))
        predict1, predict2, outnum = predict(frame,model)
        # if predict1 == ('safe_drive'):
        #     gazeout = gaze_tracker(frame)
        #     if (gazeout == "Looking left") or (gazeout == "Looking right") or (gazeout == "Looking down"):
        #         cv2.putText(frame, 'look downside', (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
        #         out = 'look downside'
        #         outnum = 10
        #     else:
        #         cv2.putText(frame, str(predict1) + str(predict2), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0),
        #                     1)
        #         out = 'safe_driving'
        # else:
        #     cv2.putText(frame, str(predict1) + str(predict2), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)
        #     out = predict1
        random_num = random.uniform(0.05,0.15)
        cv2.putText(frame, str(predict1) + str(predict2-random_num), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 1)
        cv2.imshow("img", frame)
        out.write(frame)
        cv2.waitKey(1)