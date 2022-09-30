import os, json
import string
import re
from rank_bm25 import BM25Okapi
from fuzzy_match import match
remove_punctuation_map = dict((ord(char), " ") for char in string.punctuation)
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s
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
    text = remove_accents(text)
    text = re.sub('[0-9.] \)'," ",text)
    text = re.sub('[0-9.]\)'," ",text)
    text = text.lower()
    text = text.replace("soomg","")
    text = text.replace("vitaminad","vitamin a d")
    if "tioga" in text:
        text = "tioga"
    for c in string.punctuation:
        text = text.replace(c," ")
    text = " ".join(text.split())
    return text

with open("pill_id2name.json","r",encoding="utf-8") as fr:
    pill_id2name = json.load(fr)
text2pid = {}
corpus = []
for pill_id in pill_id2name:
    for e in pill_id2name[pill_id]:
        e_clean = clean_text(e)
        if e_clean not in text2pid:
            text2pid[e_clean] = [pill_id]
        else:
            text2pid[e_clean].append(pill_id)
        if e_clean not in corpus:
            corpus.append(e_clean)
def get_top_n(query,corpus):
    # tokenized_query = query.split(" ")
    # top_k = bm25.get_top_n(tokenized_query, corpus, n=5)
    top_k= match.extract(query, corpus, limit=4)
    return top_k

tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
pres2pid = {}
with open("corpus_test_tmp.txt","r",encoding="utf-8") as lines:
    for line in lines:
        tmp = line.strip().split("\t")
        text = tmp[2]
        pres_name = line.strip().split("\t")[1]
        # print(pres_name)
        pids = []
        text  = text.lower().split(" <sep> ")
        drug_name = []
        for t in text:
            if is_valid(t):
                drug_name.append(clean_text(t))
        for query in drug_name:
            top_k = get_top_n(query, corpus)
            # if(pres_name=="VAIPE_P_TRAIN_354.png"):
            #     print(query, top_k)
            for name,score in top_k:
                if score<0.6:
                    print(query+"-----"+name)
                    continue
                pid_list = text2pid[name]
                pids.extend(pid_list)
        pres2pid[pres_name] = list(set(pids))
with open("pres2pid_public_test.json","w",encoding="utf-8") as fw:
    json.dump(pres2pid,fw,indent=4,ensure_ascii=False)
# query = "medibogan"
# query = clean_text(query)
# tokenized_query = query.split(" ")
# print(bm25.get_top_n(tokenized_query, corpus, n=5))
