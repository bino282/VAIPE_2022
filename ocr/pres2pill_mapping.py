import os
import re
import json
import numpy as np
from unidecode import unidecode
from tqdm import tqdm
labels_map = {0:"other",1:"diagnose", 2:"drugname",3:"date",4:"usage",5:"quantity"}
def remove_weight_info(text):
    text = text.lower()
    weight_info = []
    weight_info.extend(re.findall('[0-9]+mg',text))
    weight_info.extend(re.findall('[0-9]+ mg',text))
    weight_info.extend(re.findall('[0-9]+ mcg',text))
    weight_info.extend(re.findall('[0-9]+mcg',text))
    weight_info.extend(re.findall('[0-9]+g',text))
    weight_info.extend(re.findall('[0-9]+ g',text))
    weight_info.extend(re.findall('[0-9]+ui',text))
    weight_info.extend(re.findall('[0-9]+ ui',text))
    weight_info.extend(re.findall(' [0-9]+',text))
    for w in weight_info:
        text = text.replace(w, '')
    text = re.sub(r"[^a-zA-Z0-9]+", ' ', unidecode(text))
    return text.strip()
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
def clean_text(text):
    idx = []
    idx.extend(re.findall('[0-9]+ \)',text))
    idx.extend(re.findall('[0-9]+\)',text))
    idx.extend(re.findall('[0-9.]\)',text))
    idx.extend(re.findall('[0-9.] \)',text))
    for i in idx:
        text = text.replace(i, "").lower()
    text = remove_weight_info(text)
    text = text.replace("vitamin","vitamin ")
    text = text.replace("mr","")
    text = text.replace(" tab "," ")
    text = text.replace("savi","")
    text = text.replace(" s ","")
    return text.strip()

def get_jaccard_sim(str1, str2):
    tok1 =  str1.split()
    tok1 = [e for e in tok1 if not e.isdigit()]
    tok2 =  str2.split()
    tok2 = [e for e in tok2 if not e.isdigit()]
    a = set(tok1) 
    b = set(tok2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def match(text1,text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    score = get_jaccard_sim(text1,text2)
    if (score>=0.3):
        # if text1!=text2:
        #     print(text1+"---"+text2)
        return 1
            
    # token1 = text1.split()
    # token2 = text2.split()
    # for w in token1:
    #     if not any(chr.isdigit() for chr in w):
    #         if w in token2:
    #             return 1
    return 0
from itertools import combinations
import itertools
def create_pres_matrix():
    matrix = np.zeros((107,107))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i==j:
                matrix[i][j]=1
    with open(ROOT + "datasets/train_pres2pill_label.json","r",encoding="utf-8") as fr:
        data = json.load(fr)
        for k in data:
            value = data[k]
            value = [int(e) for e in value]
            if len(value)==1:
                continue
            combine = combinations(value,2)
            combine = [e for e in combine]
            for e in combine:
                matrix[e[0]][e[1]]+=1
                matrix[e[1]][e[0]]+=1 
    return matrix
matrix = create_pres_matrix()

with open(ROOT + "datasets/pill_id2name.json","r",encoding="utf-8") as fr:
    data = json.load(fr)

def get_pid(pill_name):
    result_id= []
    name_match = []
    for pill_id in data:
        list_name = data[pill_id]
        for name in list_name:
            if match(pill_name,name):
                result_id.append(pill_id)
                name_match.append(name)
                break
    return result_id, name_match
def count(pairs):
    prev = 0
    for e in pairs:
        c = 0
        combine = combinations(list(e),2)
        combine = [e for e in combine]
        for b in combine:
            c = c + matrix[int(b[0])][int(b[1])]
        if c > prev:
            prev = c
            result = e
    return result
import pandas as pd
correct = 0
count_ = 0
train_pres2pid = {}
with open(ROOT + "datasets/diagnose_pill_id2name.json","r",encoding="utf-8") as fr:
    diagnose = json.load(fr)
from thefuzz import process
def diagnose_similar(text1,text2):
    tok1 =  text1.split()
    tok1 = [e for e in tok1]
    tok2 =  text2.split()
    tok2 = [e for e in tok2]
    a = set(tok1) 
    b = set(tok2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
def get_id_by_diagnose(text, diagnose=diagnose):
    diagnose_name = diagnose.keys()
    result = ""
    score= 0
    result_prev = ""
    for key_ in diagnose_name:
        s = diagnose_similar(text,key_)
        if s > score:
            score = s
            result = key_
            result_prev = result
    return result,score,result_prev
def get_diagnose(text):
    tmp = text.strip().split(" <SEP> ")
    diagnose_out = []
    for e in tmp:
        diagnose = pipe([e])
        class_id = diagnose[0]["label"]
        if class_id ==1 :
            diagnose_out.append(e)
    return diagnose_out
# text ="Chần đoán: E78 - Rối loạn chuyển hoá lipoprotein và tình trạng tăng lipid máu khác: (E11)TD Bệnh đái"
# key_ = get_id_by_diagnose(text)
# print(key_)
# print(diagnose[key_])
# exit()
with open(ROOT + "datasets/text_classification/corpus_test.txt","r",encoding="utf-8") as lines:
    for line in tqdm(lines):
        count_ = count_ + 1
        preds = []
        text_ori = line.strip().split("\t")[2]
        label = line.strip().split("\t")[0].split(",")
        pres_name = line.strip().split("\t")[1]
        text  = text_ori.lower().split(" <sep> ")
        # diagnose_names = get_diagnose(text_ori)
        # diagnose_names = " ".join(diagnose_names)
        # key_,score,key_prev = get_id_by_diagnose(diagnose_names.lower())
        # pid = diagnose[key_]
        # pid_prev = diagnose[key_prev]
        # pid = [str(e) for e in pid]
        # pid_prev = [str(e) for e in pid_prev]
        drug_name = []
        for t in text:
            if is_valid(t):
                drug_name.append(t)
        # print(drug_name)
        for name in drug_name:
            id_,name_match = get_pid(name)
            preds.extend(id_)
        if len(drug_name)==0: 
            print("errrrrrrrrr")
        # if(set(label).issubset(set(preds)))  or ():
        #     correct +=1
        # else:
        #     print(pres_name)
        #     print(preds)
        #     print(label)
        #     print("="*50)
        train_pres2pid[pres_name] = preds
print(correct/count_)
with open("test_pres2pid_test.json","w",encoding="utf-8") as fw:
    json.dump(train_pres2pid,fw,indent=4)