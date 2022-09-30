import os
path_pred = "runs/detect/exp/labels"
image_pred_list = os.listdir(path_pred)
image_pred_list = [e for e in image_pred_list if ".txt" in e]
image_pred_list.sort()
fw = open("results.csv","w",encoding="utf-8")
fw.write("image_name,class_id,confidence_score,x_min,y_min,x_max,y_max")
fw.write("\n")
for file_name in image_pred_list:
    with open(os.path.join(path_pred, file_name)) as lines:
        for line in lines:
            tmp = line.strip().split()
            if(float(tmp[5])<0.65):
                print(file_name)
                print(tmp)
                continue
            row = [file_name.replace(".txt",".jpg"),tmp[0],tmp[5],tmp[1],tmp[2],tmp[3],tmp[4]]
            row = ",".join(row)
            fw.write(row)
            fw.write("\n")
fw.close()