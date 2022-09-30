import os
import json
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
train_path ="/datahdd/public_train/pill/"
output_path = "datasets/VAIPE_PILL_CLS/"

def crop(img,box, image_name,idx,mode="train"):
    x,y,w,h = box["x"], box["y"], box["w"], box["h"]
    crop_img = img[y:y+h, x:x+w]
    if not os.path.exists("{}/{}/{}".format(output_path,mode,box["label"])):
        os.mkdir("{}/{}/{}".format(output_path,mode,box["label"]))
    cv2.imwrite("{}/{}/{}".format(output_path,mode,box["label"]) +"/" + image_name.split(".")[0]+ "-{}".format(idx) + ".jpg",crop_img)

label_public_train = os.listdir(train_path+"label")
        
for file_name in tqdm(label_public_train):
    if(".json" not in file_name):
        continue
    file_name = file_name.replace(".txt",".json")
    image_path = "{}/image/{}".format(train_path,file_name.replace(".json",".jpg"))
    image =  cv2.imread(image_path)
    with open("{}/label/{}".format(train_path,file_name),"r",encoding="utf-8") as fr:
        boxes = json.load(fr)
    for idx,box in  enumerate(boxes):
        crop(image,box,file_name.replace(".json",".jpg"),idx,"train")

public_test_old_path = "/datahdd/public_test/pill/"
with open("datasets/gt.json","r",encoding="utf-8") as fr:
    gt = json.load(fr)

for e in gt["annotations"]:
    file_name = e["image_id"]+".jpg"
    image_path = "{}/image/{}".format(public_test_old_path,file_name)
    image =  cv2.imread(image_path)
    box = {
        "x": e["bbox"][0],
        "y": e["bbox"][1],
        "w": e["bbox"][2],
        "h": e["bbox"][3],
        "label": e["category_id"]
    }
    idx = e["id"]
    crop(image,box,file_name,idx,"val")
    
        