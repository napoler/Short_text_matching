# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
compute_kl_loss


R-Drop: Regularized Dropout for Neural Networks
This repo contains the code of our NeurIPS-2021 paper, R-drop: Regularized Dropout for Neural Networks.

R-Drop is a simple yet very effective regularization method built upon dropout, by minimizing the bidirectional KL-divergence of the output distributions of any pair of sub models sampled from dropout in model training.

@inproceedings{liang2021rdrop,
  title={R-Drop: Regularized Dropout for Neural Networks},
  author={Liang, Xiaobo* and Wu, Lijun* and Li, Juntao and Wang, Yue and Meng, Qi and Qin, Tao and Chen, Wei and Zhang, Min and Liu, Tie-Yan},
  booktitle={NeurIPS},
  year={2021}
}


https://github.com/dropreg/R-Drop


"""

import torch.nn.functional as F


# define your task model, which outputs the classifier logits
# model = TaskModel()


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


if __name__ == '__main__':
    from transformers import BertTokenizer, BertModel

    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = ["hell word !", "hdsadell wdord !", "hell asasdword !", "heldddl word !"]
    inputs = tokenizer(text, max_length=128, pad_to_max_length=True, return_tensors='pt')
    # print("inputs",inputs)
    outputs = model(**inputs)
    # keep dropout and forward twice
    logits = model(**inputs)[0]

    logits2 = model(**inputs)[0]

    # cross entropy loss for classifier
    # ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

    kl_loss = compute_kl_loss(q=logits, p=logits2)

    # carefully choose hyper-parameters
    # loss = ce_loss + α * kl_loss
    print(kl_loss)
