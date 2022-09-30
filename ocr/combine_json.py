import json
all_json = {}
with open("train_pres2pid_final.json","r",encoding="utf-8") as fr:
    train = json.load(fr)
    for k in train:
        all_json[k] = train[k]

with open("test_pres2pid_final.json","r",encoding="utf-8") as fr:
    test = json.load(fr)
    for k in test:
        all_json[k] = test[k]

with open("pres2pid_mapping.json","w",encoding="utf-8") as fw:
    json.dump(all_json,fw, indent=4)