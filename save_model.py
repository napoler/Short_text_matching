"""

save model from checkpoint
Saving & Loading Model for Inference
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
import torch

from model.rdropModel import ShortMatch
from transformers import BertTokenizerFast, BertModel

resume_from_checkpoint = input("resume_from_checkpoint path:")
PATH = "data/model/pytorch_model.bin"
# path = input("base model path:")
model = ShortMatch.load_from_checkpoint(checkpoint_path=resume_from_checkpoint)
model.freeze()
model.eval()
model.save_hyperparameters("data/model/myconfig.yaml")
torch.save(model.state_dict(), PATH)


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    pass
