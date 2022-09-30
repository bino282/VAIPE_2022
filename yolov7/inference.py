from detect_img import detect
from models.experimental import attempt_load
from utils.torch_utils import select_device
img_path = "/datahdd/nhanv/git/VAIPE_AI/datasets/public_test/pill/image/VAIPE_P_0_0.jpg"
device = select_device("0") # cuda device, i.e. 0 or 0,1,2,3 or cpu
# Load yolo model
yolo_weights = "/datahdd/nhanv/git/VAIPE_AI/yolov7/runs/train/yolov7-e6e/weights/best.pt"
yolo_model = attempt_load(yolo_weights, map_location=device)  # load FP32 model

text_arr = detect(img_path,model=yolo_model)
print(text_arr)

# load classification model
