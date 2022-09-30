# Installation
## Clone repository
~~~
https://github.com/superkido511/VAIPE_AI.git
git checkout -b release
~~~

*Build*
~~~
cd VAIPE_AI
docker build -t ai4vn:latest -f Dockerfile .
~~~

- Chuẩn bị `public_test` data vào `DATASET_PATH` theo cấu trúc `DATASET_PATH/public_test`.

- Chuẩn bị `public_train` data vào `DATASET_PATH` theo cấu trúc `DATASET_PATH/public_train`. Download các pretrained models tại link https://drive.google.com/file/d/12YqI_9cZa3E-0Cymw9IWa_P6miL1NpYx/view?usp=sharing và giải nén thư mục `models` và `DATASET_PATH` theo cấu trúc `DATASET_PATH\models`.

*Run với GPU*

~~~
docker run --gpus all -d -it --name ai4vn-ntq -v {$DATASET_PATH}:/app/datasets  ai4vn:latest
~~~


*Login vào docker*
~~~
docker container exec -it ai4vn-ntq /bin/bash
~~~

# Training

*Tạo datasets để training yolov7 và beit*
~~~
cd /app/data_processing
python create_data_classification.py
python clear_yolo_training.py 
~~~

*Train yolov7*

~~~
cd /app/yolov7
bash scripts/train.sh
~~~

Sau khi training, model mới sẽ được tạo ra ở `runs/train/yolov7-d6/weights/best.pt`

*Train beit*

~~~
cd /app/beit
bash scripts/train.sh
~~~

Sau khi training, model mới sẽ được tạo ra ở `outputs/VAIPE_PILL_108_recall_loss/beit_large_patch16_224_pt22k_ft22k/checkpoint-best.pth`

# Inference: 


*Trích xuất ID thuốc tương ứng với từng ảnh*
~~~
cd /app/ocr/
python ocr_pres.py
python tfidf_mapping.py
~~~

*Detect thuốc trong ảnh*

~~~
cd /app/yolov7
bash scripts/detect.sh
python3 create_result.py
~~~

Để sử dụng model đã được train lại, thay path model ở detect.sh

Sau khi chạy các lệnh trên, ta sẽ có được file `results.csv` chứa các tọa độ của thuốc trong ảnh

*Phân loại thuốc thuốc trong ảnh*
~~~
cd /app/beit
bash scripts/inference.sh
cd /app/data_processing
python3 post_processing.py
~~~

Để sử dụng model đã được train lại, thay path model ở inference.sh

Sau khi chạy các lệnh trên, ta sẽ có được file `results.csv` chứa các tọa độ của thuốc trong ảnh, cùng với label tương ứng.


