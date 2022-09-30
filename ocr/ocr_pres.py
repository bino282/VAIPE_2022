import os
import pandas as pd
# import seaborn as sns
import numpy as np
import json
from thefuzz import process
from thefuzz import fuzz
import re
from tqdm import tqdm
from copy import deepcopy
import shutil
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (50,20)
from PIL import Image
import cv2
from tqdm import tqdm
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import craft_text_detector
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os
import re
from unidecode import unidecode

config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = 'transformerocr.pth' # https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

root = '/app/'
output_dir = '/app/outputs'
crop_dir = output_dir + '/image_crops/'
join = os.path.join
# pres_dir = join(root,'datasets', 'public_train/prescription/label')
# pres_files = [join(pres_dir, x) for x in os.listdir(pres_dir)]

def is_valid(r):
    valid = False
    if len(re.findall('[0-9.]\)',r)) > 0 \
    or len(re.findall('[0-9.] \)',r)) > 0 \
    or len(re.findall('"[0-9]\)',r)) > 0 \
    or len(re.findall('"[0-9] \)',r)) > 0 \
    or len(re.findall('[0-9]+mg',r)) > 0 \
    or len(re.findall('[0-9]+ mg',r)) > 0 \
    or len(re.findall('[0-9]+ mcg',r)) > 0 \
    or len(re.findall('[0-9]+mcg',r)) > 0 \
    or len(re.findall('[0-9]+g',r)) > 0 \
    or len(re.findall('[0-9]+ g',r)) > 0:
        idx = re.findall('[0-9.]\)',r)
        idx.extend(re.findall('[0-9.] \)',r))
        idx.extend(re.findall('"[0-9.]\)',r))
        idx.extend(re.findall('"[0-9.] \)',r))
        idx.extend(re.findall('[0-9]+\)',r))
        if len(idx) == 0:
            return valid
        for i in idx:
            if r.startswith(i):
                valid = True
                break
    return valid


def extract_pill_text(img_path, craft_net, refine_net,debug=False):
    image = read_image(img_path)
    prediction_result = craft_text_detector.get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )
    crop_dir = output_dir + '/image_crops/'
    shutil.rmtree(crop_dir, ignore_errors=True)
    os.mkdir(crop_dir)
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=output_dir,
        rectify=True
    )
    results = []
    for p in os.listdir(crop_dir):
        img_p = crop_dir + p
        if not img_p.endswith('.png'):
            continue
        img = read_image(img_p)
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10,cv2.BORDER_CONSTANT, None, value=0)
        r = detector.predict(Image.fromarray(img))
        # if not is_valid(r):
        #     continue
        results.append(r)
    return results
if __name__ == '__main__':
    test_path = "/app/datasets/public_test/"
    list_file = os.listdir(os.path.join(test_path,"prescription/image"))
    fw = open("corpus_test_tmp.txt","w",encoding="utf-8")
    for pres_name in tqdm(list_file):
        if ".png" not in pres_name:
            continue
        try:
            pres_path = os.path.join(os.path.join(test_path,"prescription/image"),pres_name)
            pill_name = extract_pill_text(pres_path,craft_net,refine_net)
            text = " <SEP> ".join(pill_name)
            label = ["None"]
            fw.write("\t".join([",".join(label),pres_name,text]))
            fw.write("\n")
        except Exception as e:
            print(e)
            print(pres_path)
    fw.close()
