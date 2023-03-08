import cv2
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression,  scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device,  time_synchronized, TracedModel

np.random.seed(10)

def load_image(img0, img_size, stride):
    img = letterbox(img0, img_size, stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    return img, img0


def load_model(weights, param_device, image_s, trace):
    device = select_device(param_device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(image_s, s=stride)
    if trace:
        model = TracedModel(model, device, image_s)
    if half:
        model.half()  # to FP16

    return model, stride, imgsz, half

def detect(dict_params, model, stride, imgsz, half, image):

    device = select_device(dict_params["device"])

    img, im0s = load_image(image, dict_params["img_size"], stride)
    
    if device.type != 'cpu':  
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=dict_params["augment"])[0]

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=dict_params["augment"])[0]
        
    # Apply NMS
    pred = non_max_suppression(pred, dict_params["conf_thres"], dict_params["iou_thres"], classes=dict_params["classes"], agnostic=dict_params["agnostic_nms"])

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        im0 = im0s

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):

                label = f'{class_names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    return im0



dict_params = {
    "device": 'cpu',
    "weights": "./ckpt/yolov7.pt",
    "img_size": 640,
    "classes": None,

    "source": "image.jpg",
    "augment": False,
    
    "conf_thres": 0.25,
    "iou_thres":0.45,

    "project": "runs/custom_detect",
    "name":"predict_video",
    "agnostic_nms":True, 
    "view_img": True,
    "save_img": False,
    "save_txt": True,
    "save_conf": False,
    "trace": True,
    "exist_ok": False,
}


model, stride, imgsz, half = load_model(dict_params["weights"], dict_params["device"], dict_params["img_size"], dict_params["trace"])

class_names = model.module.names if hasattr(model, 'module') else model.names
colors = [(255, 0, 0), (0, 255, 0), (0, 0,255)]

vid_path = "./trafic.mp4"
vid = cv2.VideoCapture(vid_path)

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while vid.isOpened():
    ret, frame = vid.read()
    
    if ret:
    
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        
        img = detect(dict_params, model, stride, imgsz, half, frame)

        cv2.putText(frame,fps, (10,50), font,  1, (255,255,255), 1, 2)
        cv2.imshow("frame", img)

        k = cv2.waitKey(1) & 0xff 
        if k == 27: 
            break 
    else:
        break

vid.release()
cv2.destroyAllWindows() 



