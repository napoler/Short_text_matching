from transformers import pipeline
# unmasker = pipeline('fill-mask', model='napoler/chinese_roberta_L-4_H-512_rdrop')
unmasker = pipeline('fill-mask', model='data/model')
print(unmasker("中国的首都是[MASK]京。"))