import pandas as pd
gt_df = pd.read_csv("datasets/gt.csv")
gt_img_id = {}
for idx, row in gt_df.iterrows():
    image_name = row.image_name
    if image_name not in gt_img_id:
        gt_img_id[image_name] = [row.class_id]
    else:
        gt_img_id[image_name].append(row.class_id)

pred_df = pd.read_csv("results.csv")
pred_img_id = {}
for idx, row in pred_df.iterrows():
    image_name = row.image_name
    if image_name not in pred_img_id:
        pred_img_id[image_name] = [row.class_id]
    else:
        pred_img_id[image_name].append(row.class_id)
def check(e1,e2):
    if len(e1)!=len(e2):
        return 0
    for e in e1:
        if e not in e2:
            return 0
    for e in e2:
        if e not in e1:
            return 0
    return 1
for image_name in gt_img_id:
    try:
        id_gt = gt_img_id[image_name]
        pred_gt = pred_img_id[image_name]
        if check(id_gt,pred_gt):
            continue
        else:
            print(image_name)
    except:
        continue