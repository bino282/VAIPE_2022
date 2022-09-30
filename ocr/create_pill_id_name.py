import os
import json
import re
import copy
def clean_text(text):
    idx = []
    idx.extend(re.findall('[0-9]+ \)',text))
    idx.extend(re.findall('[0-9]+\)',text))
    idx.extend(re.findall('[0-9.]\)',text))
    idx.extend(re.findall('[0-9.] \)',text))
    for i in idx:
        text = text.replace(i, "").lower()
    text = text.replace("(","")
    text = text.replace(")","")
    return text.strip().lower()
pres_lis = os.listdir("/datahdd/public_train/prescription/label")
id2name = {}
diagnose_pill_id2name = {}
for file_name in pres_lis:
    if ".json" not in file_name:
        continue
    with open("/datahdd/public_train/prescription/label/"+file_name, "r") as fr:
        data = json.load(fr)
        diagnose = ""
        for e in data:
            if e["label"]=="diagnose":
                diagnose += e["text"]
        diagnose= diagnose[11:]
        diagnose = diagnose.strip().lower()
        
        if diagnose not in diagnose_pill_id2name:
            diagnose_pill_id2name[diagnose]=[]
        for e in data:
            if e["label"]=="drugname":
                text_clean = clean_text(e["text"])
                if e["mapping"] not in diagnose_pill_id2name[diagnose]:
                    diagnose_pill_id2name[diagnose].append(e["mapping"])
                if e["mapping"] not in id2name:
                    id2name[e["mapping"]] = {text_clean:1}
                else:
                    if text_clean not in id2name[e["mapping"]]:
                        id2name[e["mapping"]][clean_text(e["text"])]=1
                    else:
                        id2name[e["mapping"]][clean_text(e["text"])]+=1

# def check_k(k,cate,id2name):
#     for e in id2name:
#         for key_ in id2name[e]:
#             if k ==key_ and e!= cate:
#                 return 1
#     return 0
# origin_dict = copy.deepcopy(id2name)
# for e in id2name:
#     value = id2name[e]
#     if len(value)>1:
#         for k in value:
#             if value[k]<3:
#                 if check_k(k,e,id2name):
#                     print(k)
#                     print(e)
#                     del origin_dict[e][k]
with open("pill_id2name.json","w",encoding="utf8") as fw:
    json.dump(id2name, fw,indent=3)

# with open("diagnose_pill_id2name.json","w",encoding="utf8") as fw:
#     json.dump(diagnose_pill_id2name, fw,indent=3,ensure_ascii=False)