import os
import json
with open("/datahdd/public_test/pill_pres_map.json","r",encoding="utf-8") as fr:
    pill_pres_map = json.load(fr)

with open(ROOT + "datasets/gt.json","r",encoding="utf-8") as fr:
    gt = json.load(fr)

pill2id  = {}
for e in gt["annotations"]:
    img_id  = e["image_id"]+".jpg"
    if img_id not in pill2id:
        pill2id[img_id] = [str(e["category_id"])]
    else:
        pill2id[img_id].append(str(e["category_id"]))
print(pill2id)

pres2pid = {}
for e in pill_pres_map:
    pres_name = e["pres"].replace(".json",".png")
    if pres_name not in pres2pid:
        pres2pid[pres_name] = []
    for pill_name in e["pill"]:
        pill_ids  = pill2id[pill_name.replace(".json",".jpg")]
        for id_ in pill_ids:
            if id_ not in pres2pid[pres_name] and id_!="107":
                pres2pid[pres_name].append(id_)

with open("gt_public_test_pres.json","w",encoding="utf-8") as fw:
    json.dump(pres2pid,fw,indent=4)
