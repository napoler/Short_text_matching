# 预处理数据集

# /run/media/terry/data/dev/tdata/web_text_zh_valid.json

import os
import json
import csv
# from re import split
from tqdm.auto import tqdm
file="/run/media/terry/data/dev/tdata/web_text_zh_train.json"
out_file="data/out_data.csv"
datas=[]
with open(file,'r') as f:
    for i,line in enumerate( tqdm(f)):
        # print(line)
        item=json.loads(line)
        datas.append([item['title']])
        for l in item['desc'].replace("？","。").replace("\n","。").replace("\r","。").split("。"):
            if len(l)>4:
                datas.append([l])

        # datas.extend(item['desc'].split("。"))
        if i>150000:
            break

print(datas[:10])
print(len(datas))
with open(out_file,'w') as f:
    write=csv.writer(f)
    write.writerows(datas)
    # print(dir(write))

