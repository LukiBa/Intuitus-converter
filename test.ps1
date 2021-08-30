conda activate nn
python test_torch_yolo.py --batch-size 8 --wdir ./weights --weights last.pt --cfg  torch_yolo/cfg/yolov3tiny/yolov3-tiny-relu.cfg --data torch_yolo/data/coco2017.data --quantized 0