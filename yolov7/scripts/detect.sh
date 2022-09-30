rm -rf outputs
mkdir outputs
python detect.py --weights ../datasets/models/yolov7.pt \
                --conf 0.6 --img-size 1280 \
                --iou-thres 0.45 \
                --device 1 \
                --source ../datasets/public_test/pill/image \
                --save-txt --save-conf
