import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from torchmetrics.functional import accuracy

# from ..SentenceTransformer import SentenceTransformer
# from .. import util
# def cos_sim()
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    # print("token_embeddings",token_embeddings)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def similarity_fct(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])

    https://github.com/UKPLab/sentence-transformers/blob/d5b011583b8d689591d52b0d244be315f2800d30/sentence_transformers/util.py#L23
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.
        The performance usually increases with increasing batch sizes.
        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)
        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
        Example::
            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader
            from transformers import BertTokenizer, BertForSequenceClassification,BertModel
            model = BertModel.from_pretrained('bert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """

    def __init__(self, model, scale, similarity_fct=None):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
         dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        """


        https://colab.research.google.com/drive/1BVZWhidtgsUH-vK-LcMlFB0jbS-RebKh#scrollTo=6y7fkJ99GQsn&line=2&uniqifier=1
        :param sentence_features:
        :param labels:
        :return:
        """
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        if type(sentence_features) == tuple and len(sentence_features) > 1:

            reps = [mean_pooling(self.model(input_ids=input_ids, attention_mask=attention_mask)[0], attention_mask) for
                    input_ids, attention_mask in sentence_features]
            embeddings_a = reps[0]  # B*D
            embeddings_b = torch.cat(reps[1:])  # (B-1)*B*D
        else:
            input_ids, attention_mask = sentence_features
            embeddings_a = mean_pooling(self.model(input_ids=input_ids, attention_mask=attention_mask)[0],
                                        attention_mask)  # B*D
            embeddings_b = mean_pooling(
                self.model(input_ids=input_ids, attention_mask=attention_mask)[0], attention_mask)  # (B-1)*B*D

        if self.similarity_fct is None:
            scores = similarity_fct(embeddings_a, embeddings_b) * self.scale
        else:
            scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # print("scores",scores.argmax(-1))
        # print("scores", embeddings_a.size(),embeddings_b.size(), scores.size())

        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]
        acc = accuracy(scores.argmax(-1).view(-1), labels.view(-1).long())
        # print("acc",acc)
        # print("scores, labels",scores, scores)

        return self.cross_entropy_loss(scores, labels),acc

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


# from transformers import BertTokenizer, BertForSequenceClassification
# MultipleNegativesRankingLoss()
if __name__ == "__main__":
    from transformers import BertTokenizer, BertForSequenceClassification, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(["Hello, my dog is cute"] * 5, return_tensors="pt")
    # labels = torch.tensor([1]).unsqueeze(0) # Batch size 1
    # outputs = model(**inputs)
    # # loss = outputs.loss
    # logits = outputs[0]
    #

    sim = nn.CosineSimilarity()
    ls = MultipleNegativesRankingLoss(model=model, scale=20, similarity_fct=sim)

    ls(sentence_features=inputs['input_ids'], labels=inputs['input_ids'])
