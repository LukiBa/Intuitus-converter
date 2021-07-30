# Intuitus converter
![GitHub](https://img.shields.io/github/license/LukiBa/Intuitus-converter)
Quantization aware training and CNN model converter for FPGA based Intuitus hardware accelerator.

Quantization aware training is based on https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone.git. 

## Features
- [x] Training and testing multiple Yolov3 or Yolov4 implementations 
- [x] Quantization-aware training
- [x] Testing expected FPGA behaviour using Torch model with GPU supported computations 
- [x] Converting Torch model into Keras 
- [x] Post-training quantization using tensorflow quantizer
- [x] Translation of quantized Torch or Keras models into FPGA interpretable commands

## Supported Layers
- [x] conv2d (kernel sizes: [1x1,3x3,5x5]; strides: [1,2])
- [x] inplace maxpool2d (stride 2 only)
- [x] maxpool2d
- [x] upsample
- [x] concat
- [x] split
- [ ] inplace yolo layer  
- [ ] fully connected
- [ ] inverse bottleneck 
- [ ] residual 

## Installation
````sh
cd Intuitus-converter
pip install -e .
````

## Workflow:
1.  Select a pretrained model and download pretrained weights
2.  Use train\_torch\_yolo.py for quantization aware training
3.  Fuse batchnormalization layer and retrain till convergence if neccessary (set quantize 0, FPGA = False)
4.  Change activation to ReLU by editing cfg file (neccessary for FPGA) and retrain till convergence (keep quantize = 0)
5.  Set quantize to 1 and retrain till convergence
6.  Use torch\_convert\_postscale.py to fuse quantization scaling to a single scale (shift)
7.  Set quantize to 2 and test if mAP has changed to to scale fusing (should not have changed)
8.  Set FPGA = True to test which mAP can be achieved using the Intuitus hardware accelerator 
9.  Generate commands for the Intuitus hardware accelerator using generate\_intuitus\_from\_torch.py
10. Test the results on hardware following the steps in the related projects

## Related Projects: 
| Project | link |
| ------ | ------ |
| Intuitus Interface | <https://github.com/LukiBa/Intuitus-intf.git> |
| Yolov3-tiny example application for Zybo-Z7-20 board | <https://github.com/LukiBa/zybo_yolo.git> |
| Intuitus device driver | link to kernel module comming soon (contact author for sources) |
| Vivado Example project| https://github.com/LukiBa/zybo_yolo_vivado.git |
| Intuitus FPGA IP | encrypted trial version comming soon (contact author) |

## Author
Lukas Baischer 
lukas_baischer@gmx.at