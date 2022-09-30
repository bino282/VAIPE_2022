python train_aux.py --workers 8 --device 1 \
                    --batch-size 4 --data data/vaipe_full.yaml \
                    --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml \
                    --single-cls \
                    --cache-images \
                    --resume \
                    --weights '../datasets/models/yolov7-d6_training.pt' --name yolov7-d6 --hyp data/hyp.scratch.custom.yaml
