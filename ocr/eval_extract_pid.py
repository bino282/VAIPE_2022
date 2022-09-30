import json
with open("gt_public_test_pres.json","r",encoding="utf-8") as fr:
    gt = json.load(fr)
with open("pres2pid_public_test.json","r",encoding="utf-8") as fr:
    pred = json.load(fr)
count = 0
correct = 0
def is_subset(list1, list2):
    for e in list1:
        if e not in list2:
            return 0
    return 1
for pres_name in pred:
    pid_pred = pred[pres_name]
    pid_lb = gt[pres_name]
    count += 1
    if len(pid_pred)>0:
        if is_subset(pid_lb,pid_pred):
            correct+=1
        else:
            print(pres_name)
            print(pid_pred)
            print(pid_lb)
    else:
        print(pres_name)
print(correct/count)