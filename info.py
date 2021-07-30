"""
Based on: "https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone.git" by SpursLipu
"""
import argparse
from torch_yolo.models import *
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='torch_yolo/cfg/yolov3tiny/yolov3-tiny-quant.cfg', help='*.cfg path')
parser.add_argument('--img_size', type=int, default=416, help='img_size')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
opt = parser.parse_args()

device = torch_utils.select_device(opt.device)
model = Darknet(opt.cfg)
# model.fuse()
model.to(device)
summary(model, input_size=(3, opt.img_size, opt.img_size))
