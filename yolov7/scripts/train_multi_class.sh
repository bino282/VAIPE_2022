python train_aux.py --workers 1 --device 0 \
                    --batch-size 4 --data data/vaipe_full.yaml \
                    --img 1280 1280 --cfg cfg/training/yolov7-d6.yaml \
                    --cache-images \
                    --weights 'weights/yolov7-d6_training.pt' --name yolov7-d6-multi-cls --hyp data/hyp.scratch.custom.yaml