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
text = ["4余万字新译著将出版"]
encoded_input = tokenizer(text, return_tensors='pt',padding=True,truncation=True )
output = model(**encoded_input)
out=mean_pooling(output['last_hidden_state'],encoded_input['attention_mask'])



full="""
金晓宇翻译的部分作品
40余万字新译著将出版
金晓宇的最新译著是德国思想家本雅明的书信集，这部书稿凝结了他的很多心血。记者从出版方了解到，《本雅明书信集》目前处于出版流程中，由上海人民出版社·光启书局及行思文化一起编辑推进，力争尽早出版。这本书体量较大，难度较高，总篇幅40余万字，“我们扎扎实实做好这本书，希望不负读者的期待。”
这部新译著《本雅明书信集》收录德国思想家本雅明信件300多封，展现了本雅明渊博的知识和独特的文笔，对理解20世纪前半期欧洲文化和思想人物有特殊的参考价值。
在豆瓣《本雅明书信集》条目下，已有将近1500名读者表示“想读”，不少读者还留言表达对这本书的期待——“感谢金晓宇先生，期待读到这本书，致敬生活的勇气！”“向译者及家人致敬”“与《美丽心灵》的主人公一样，我们看到了一个中国普通家庭存在的意义，并激励了很多年轻人”……
获得最多点赞的是这两条留言——
“文字会记住你，书本上永远有你的注脚。”
“抛开痛苦的喧嚣的一切，这是一本动人的书。不要打扰译者和他的家人。”
综合：澎湃新闻、上观新闻
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


