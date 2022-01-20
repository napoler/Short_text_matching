import torch
from transformers import pipeline
from transformers import BertTokenizer, BertModel
# unmasker = pipeline('fill-mask', model='napoler/chinese_roberta_L-4_H-512_rdrop')
unmasker = pipeline('fill-mask', model='data/model')
# print(unmasker("中国的首都是[MASK]京。"))

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    # print("token_embeddings",token_embeddings)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




tokenizer = BertTokenizer.from_pretrained('data/model')
model = BertModel.from_pretrained("data/model")
model.eval()
text = ["为什么借款后一直没有给我回拨电话"]
encoded_input = tokenizer(text, return_tensors='pt',padding=True,truncation=True )
output = model(**encoded_input)
out=mean_pooling(output['last_hidden_state'],encoded_input['attention_mask'])



full="""
怎么最近安全老是要改密码呢好麻烦
你好 我昨天晚上申请的没有打电话给我 今天之内一定会打吗
我的额度多少钱
怎么申请借款后没有打电话过来呢！
"""

textB = full.split("\n")
encoded_input = tokenizer(textB, return_tensors='pt',padding=True,truncation=True )
output = model(**encoded_input)
outB=mean_pooling(output['last_hidden_state'],encoded_input['attention_mask'])




# print(out)
# print(outB)



from torch import  nn

cos=nn.CosineSimilarity()

sim=cos(out,outB)
# print(cos(out,outB))
# output[""]


for i,(rank,it) in enumerate(zip(sim.tolist(),full.split("\n"))):
    print(i,rank,it)


