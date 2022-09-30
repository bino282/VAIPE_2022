import json
import os
import os.path as osp
from PIL import Image, ExifTags
from tqdm import tqdm
import shutil
import pandas as pd
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
import os
train_list = os.listdir("/datahdd/public_train/pill/image")
train = [e for e in train_list if ".jpg" in e]
val_list = os.listdir("/datahdd/public_test/pill/image")
val =  [e for e in val_list if ".jpg" in e]
df_box_val = pd.read_csv("datasets/gt.csv")
img2box = {}
for idx, row in df_box_val.iterrows():
    img_name = row.image_name
    box  = {'label':int(row.class_id),'x':int(row.x_min),'y':int(row.y_min),'w':int(row.x_max) - int(row.x_min),'h':int(row.y_max) - int(row.y_min)}
    if img_name not in img2box:
        img2box[img_name] = [box]
    else:
        img2box[img_name].append(box)
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

dataset_path = "datasets/VAIPE_yolo"
if not osp.exists(dataset_path):
    os.makedirs(dataset_path)
    os.makedirs(osp.join(dataset_path, "images"))
    for s in ["train", "val"]:
        os.makedirs(osp.join(dataset_path, "images", s))
        os.makedirs(osp.join(dataset_path, "labels", s))
for set_data in ["val"]:
    if set_data == "train":
        file_list = train
        path  = "/datahdd/public_train/pill/"
    else:
        file_list = val
        path  = "/datahdd/public_test/pill/"
    for file in tqdm(file_list):
        if file.endswith(".jpg"):
            if set_data == "train":
                file = file.replace(".jpg",".json")
                with open(osp.join(path, "label", file)) as f:
                    data = json.load(f)
                    image_path = osp.join(path, "image", file.replace(".json", ".jpg"))
                    img = Image.open(image_path)
                    shutil.copy(image_path, osp.join(dataset_path, "images", set_data, file.replace(".json", ".jpg")))
                    w,h = exif_size(img)
                    for box in data:
                        label = box['label']
                        x_mid = (box["x"]/w + box["x"]/w + box["w"]/w)/2
                        y_mid = (box["y"]/h + box["y"]/h + box["h"]/h)/2
                        w_norm = box["w"]/w
                        h_norm = box["h"]/h
                        if x_mid > 1 or y_mid > 1 or w_norm>1 or h_norm>1:
                            print(image_path.split("/")[-1])
                            print(file.replace(".json", ".txt"))
                            print(img.size)
                            print(box)
                            continue
                        with open(osp.join(dataset_path, "labels", set_data ,file.replace(".json", ".txt")), 'a') as f:
                            f.write(f'{label} {x_mid} {y_mid} {w_norm} {h_norm}\n')
            else:
                image_path = osp.join(path, "image", file)
                img = Image.open(image_path)
                shutil.copy(image_path, osp.join(dataset_path, "images", set_data, file.replace(".json", ".jpg")))
                w,h = exif_size(img)
                data = img2box[file]
                for box in data:
                    label = box['label']
                    x_mid = (box["x"]/w + box["x"]/w + box["w"]/w)/2
                    y_mid = (box["y"]/h + box["y"]/h + box["h"]/h)/2
                    w_norm = box["w"]/w
                    h_norm = box["h"]/h
                    if x_mid > 1 or y_mid > 1 or w_norm>1 or h_norm>1:
                        print(image_path.split("/")[-1])
                        print(file.replace(".json", ".txt"))
                        print(img.size)
                        print(box)
                        continue
                    with open(osp.join(dataset_path, "labels", set_data ,file.replace(".jpg", ".txt")), 'a') as f:
                        f.write(f'{label} {x_mid} {y_mid} {w_norm} {h_norm}\n')

        else:
            print("no json")
            break
