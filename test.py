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
text = ["也认识了他的父亲、母亲","金晓宇的最新译著是德国思想家本雅明的书信集，这部书稿凝结了他的很多心血。"]
encoded_input = tokenizer(text, return_tensors='pt',padding=True,truncation=True )
output = model(**encoded_input)
out=mean_pooling(output['last_hidden_state'],encoded_input['attention_mask'])


textB = ["在豆瓣《本雅明书信集》条目下，已有将近1500名读者表示“想读”，不少读者还留言表达对这本书的期待——“感谢金晓宇先生，者及家人致敬”“与《美丽心灵》的主人公一样，我们看到了一个中国普通家庭存在的意义，并激励了很多年轻人”……、母亲",
         "金晓宇的最新译著是德国思想家的书书稿凝结了他的很多心血。"]
encoded_input = tokenizer(textB, return_tensors='pt',padding=True,truncation=True )
output = model(**encoded_input)
outB=mean_pooling(output['last_hidden_state'],encoded_input['attention_mask'])




print(out)
print(outB)



from torch import  nn

cos=nn.CosineSimilarity()


print(cos(out,outB))
# output[""]