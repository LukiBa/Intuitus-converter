conda activate nn
python train_torch_yolo.py --epochs 1 --batch-size 8 --wdir ./weights --weights yolov3-tiny.weights --data torch_yolo/data/coco2017.data --quantized 0