import os
import pandas as pd
import seaborn as sns
import numpy as np
import json
from thefuzz import process
from thefuzz import fuzz
import re
from copy import deepcopy
import shutil
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (50,20)
from PIL import Image
import cv2
from tqdm import tqdm
import os
import re
from unidecode import unidecode
join = os.path.join

with open('/app/VAIPE_AI/ocr/pres2pid_public_test_new.json') as json_file:
    pres2pid = json.load(json_file)


def postprocess_results(results_path, out_path='results.csv'):
    try:
        global pres2pid
        results_df = pd.read_csv(results_path)
        img_files = list(results_df['image_name'].values)
        img_jsons = [x.replace('.jpg', '.json') for x in img_files]
        img2pres_dict = {}
        for img_path in img_jsons:
            pres_id = re.findall('[0-9]+',img_path)[0]
            pres_path = 'VAIPE_P_TEST_NEW_{}.json'.format(pres_id)
            img2pres_dict[img_path] = pres_path

        results_df['pres_json'] = results_df['image_name'].map(lambda x: img2pres_dict[x.replace('.jpg', '.json')])
        results_df['pres_image'] = results_df['pres_json'].map(lambda x: x.replace('.json', '.png'))

        for i in range(len(results_df)):
            # data = results_df.at[i]
            image_path = results_df.at[i, 'image_name']
            pred = results_df.at[i, 'class_id']
            confident = results_df.at[i, 'confidence_score']
            pres_img = results_df.at[i, 'pres_image']
            pres_ids = pres2pid[pres_img]
            # topk = results_df.at[i, 'topk']
            # topk_prob = results_df.at[i, 'topk_prob']
            # topk = topk.split()
            # topk_prob = topk_prob.split()
            # topk_prob = [float(e) for e in topk_prob]
            if (str(pred) not in pres_ids):
                pred = '107'
                confident = 1.0
            results_df.at[i, "class_id"] = pred
            results_df.at[i, "confidence_score"] = confident

        results_df = results_df[['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max',
               'y_max']]

        results_df.to_csv(out_path,index=False)
        return True
    
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    postprocess_results("/app/VAIPE_AI/output_results/beit/results.csv")
