"""
MLM训练

"""

import pytorch_lightning as pl
import torch
# from torch.utils.data import
# from performer_pytorch import PerformerLM
from torch import nn
# from pytorch_lightning.metrics import functional as FM
# from torchvision import transforms
# from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from transformers import AdamW, BertConfig, BertModel, BertTokenizer

try:
    from .compute_kl_loss import compute_kl_loss
    from .MultipleNegativesRankingLoss import MultipleNegativesRankingLoss, cos_sim
except:
    # pass
    try:
        from compute_kl_loss import compute_kl_loss
        from MultipleNegativesRankingLoss import MultipleNegativesRankingLoss, cos_sim
    except:
        pass

torch.multiprocessing.set_sharing_strategy('file_system')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    # print("token_embeddings",token_embeddings)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def collate_fn_del(batch):
    """
    remove de
    :param batch:
    :return batch : 新数据 [batch+[idx, out]]
    """
    input_ids, attention_mask, input_ids_b, attention_mask_b, labels = [], [], [], [], []
    # print("input_ids, attention_mask, input_ids_b, attention_mask_b, labels", input_ids, attention_mask, input_ids_b,
    #       attention_mask_b, labels)
    # tmp = []
    # print("batch",len(batch[0]))
    if len(batch[0]) > 2:
        for input_ids_one, attention_mask_one, input_ids_b_one, attention_mask_b_one, labels_one in batch:

            if input_ids_b in input_ids:
                continue

            # tmp.append(input_ids_one.tolist())
            input_ids.append(input_ids_one.tolist())
            attention_mask.append(attention_mask_one.tolist())
            input_ids_b.append(input_ids_one.tolist())
            attention_mask_b.append(attention_mask_b_one.tolist())
            labels.append(labels_one.tolist())

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(
            input_ids_b), torch.LongTensor(attention_mask_b), torch.LongTensor(labels)
    else:
        for input_ids_one, attention_mask_one in batch:
            input_ids.append(input_ids_one.tolist())
            attention_mask.append(attention_mask_one.tolist())

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)


class ShortMatch(pl.LightningModule):
    """
    基于预训练微调


    """

    def __init__(self, lr=1e-5,
                 optimizer_name="AdamW",
                 pretrained="uer/chinese_roberta_L-2_H-128",
                 dropout=0.4,
                 batch_size=32,
                 num_workers=2,
                 trainfile="./data/train.pkt",
                 valfile="./data/val.pkt",
                 testfile="./data/test.pkt", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.tokenizer = AlbertTokenizer.from_pretrained(self.hparams.pretrained)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained)
        self.config = BertConfig.from_pretrained(self.hparams.pretrained)
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout
        #         self.model =RobertaForSequenceClassification.from_pretrained("clue/roberta_chinese_pair_large")
        self.model = BertModel.from_pretrained(self.hparams.pretrained, config=self.config)


    def forward(self, input_ids, attention_mask=None, input_ids_b=None, attention_mask_b=None,
                labels=None):
        # cos_sim = cos_sim
        loss_cos = MultipleNegativesRankingLoss(self.model, scale=20)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         )
        pred1 = out[0]
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         )
        pred2 = out[0]
        kl_loss = compute_kl_loss(q=pred1, p=pred2)

        if input_ids_b is not None:

            out = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b,
                             )
            pred3 = out[0]

            out = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b,
                             )
            pred4 = out[0]

            loss1, acc1 = loss_cos(((input_ids_b, attention_mask_b), (input_ids, attention_mask)), input_ids_b)
            loss2, acc2 = loss_cos(((input_ids, attention_mask), (input_ids, attention_mask)), input_ids_b)
            loss3, acc3 = loss_cos(((input_ids_b, attention_mask_b), (input_ids_b, attention_mask_b)), input_ids_b)

            kl_loss2 = compute_kl_loss(q=pred3, p=pred4)
            kl_loss = kl_loss + kl_loss2
            loss = loss1 + loss2 + loss3
            self.log_dict({"acc3": acc3, "acc1": acc1, "acc2": acc2})
            # print({"acc3": acc3, "acc1": acc1, "acc2": acc2})
            # loss2 = loss_cos(input_ids, input_ids)

            if labels is not None:
                # return pred1, loss, loss, kl_loss / input_ids.size(0)
                return pred1, kl_loss / 2 / input_ids.size(0) + loss, loss, kl_loss / input_ids.size(0)
        else:
            loss = kl_loss
            return pred1, kl_loss / input_ids.size(0)
        return pred1

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        if len(batch) > 2:
            input_ids, attention_mask, input_ids_b, attention_mask_b, labels = batch

            out, loss, sloss, kl_loss = self(input_ids=input_ids, attention_mask=attention_mask,
                                             input_ids_b=input_ids_b,
                                             attention_mask_b=attention_mask_b, labels=labels)
        else:
            input_ids, attention_mask = batch

            out, loss = self(input_ids=input_ids, attention_mask=attention_mask)
            kl_loss = loss
            sloss = loss

        metrics = {
            # 'train_acc': acc,
            'train_loss': loss,
            'train_sloss': sloss,
            'train_kl_loss': kl_loss,

        }
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # print("len batch", batch)
        if len(batch) > 2:
            input_ids, attention_mask, input_ids_b, attention_mask_b, labels = batch

            out, loss, sloss, kl_loss = self(input_ids=input_ids, attention_mask=attention_mask,
                                             input_ids_b=input_ids_b,
                                             attention_mask_b=attention_mask_b, labels=labels)
        else:
            input_ids, attention_mask = batch

            out, loss = self(input_ids=input_ids, attention_mask=attention_mask)
            kl_loss = loss
            sloss = loss
        metrics = {
            # 'val_acc': acc,
            'val_loss': loss,
            'val_sloss': sloss,
            'val_kl_loss': kl_loss,
        }
        self.log_dict(metrics)
        return metrics

    #     def configure_optimizers(self):
    #         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #         return optimizer
    def configure_optimizers(self):
        """优化器 # 类似于余弦，但其周期是变化的，初始周期为T_0,而后周期会✖️T_mult。每个周期学习率由大变小； https://www.notion.so/62e72678923f4e8aa04b73dc3eefaf71"""
        #         optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))
        #         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        if self.hparams.optimizer_name == "AdamW":
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = getattr(torch.optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.lr)
        #         使用自适应调整模型
        T_mult = 2
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, min_lr=1.0e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=T_mult, eta_min=0,
                                                                         verbose=False)
        lr_scheduler = {
            #            'optimizer': optimizer,
            'scheduler': scheduler,
            #             'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'interval': 'step',  # epoch/step
            'frequency': 1,
            'name': "lr_scheduler",
            'monitor': 'train_loss',  # 监听数据变化
            'strict': True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        train = torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size), collate_fn=collate_fn_del,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        val = torch.load(self.hparams.valfile)
        # return DataLoader(val, batch_size=int(self.hparams.batch_size),num_workers=2,pin_memory=True)
        # return DataLoader(val, batch_size=int(self.hparams.batch_size), collate_fn=collate_fn_del,
        #                   num_workers=self.hparams.num_workers,
        #                   pin_memory=True)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), collate_fn=collate_fn_del,
                          num_workers=self.hparams.num_workers,
                          pin_memory=False)
        # # 对于使用softmax活在禁用attention_mask数据
        # if self.hparams.decode == "softmax" or self.hparams.attention_mask == False:
        #     return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=self.hparams.num_workers, pin_memory=True)
        # if self.hparams.acc:
        #     return DataLoader(val, batch_size=1, num_workers=self.hparams.num_workers, pin_memory=True)
        # else:
        #     return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test = torch.load(self.hparams.valfile)
        if self.hparams.decode == "softmax" or self.hparams.attention_mask == False:
            return DataLoader(test, batch_size=int(self.hparams.batch_size), collate_fn=collate_fn_del,
                              num_workers=self.hparams.num_workers,
                              pin_memory=True)
        # return DataLoader(test, batch_size=int(self.hparams.batch_size),num_workers=2,pin_memory=True)
        if self.hparams.acc:
            return DataLoader(test, batch_size=1, num_workers=self.hparams.num_workers, pin_memory=True)
        else:
            return DataLoader(test, batch_size=int(self.hparams.batch_size), collate_fn=collate_fn_del,
                              num_workers=self.hparams.num_workers,
                              pin_memory=True, shuffle=False)
