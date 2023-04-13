import os
import json

import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from model_cbam import efficientnet_b0 as create_model

from urllib import request


def predict(img,model_path='model/model-70_0.978_fix.pth'):
    img = cv2.resize(img, (224, 224))
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
    model = create_model(num_classes=12).to(device)
    # load model weights
    model_weight_path = model_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
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
if __name__ == '__main__':
    webcam = cv2.VideoCapture('DJI_0068.MP4')
    while True:
        status, frame = webcam.read()
        if not status:
            break
        frame = cv2.resize(frame, (224, 224))
        predict(frame)

