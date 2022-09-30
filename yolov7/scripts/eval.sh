python test.py --data data/vaipe_full.yaml \
            --img 1280 --batch 32 --conf 0.5 --iou 0.45 \
            --single-cls \
            --device 0 --weights runs/train/yolov7-d6/weights/best.pt --name yolov7_1280_val
