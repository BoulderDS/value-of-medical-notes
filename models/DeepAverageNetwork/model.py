"""
this code is modified from https://github.com/ksenialearn/bag_of_words_pytorch/blob/master/bag_of_words-master-FINAL.ipynb
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

 
class DAN(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim, feature_len, dropout, is_feature=False, is_text=True):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(DAN, self).__init__()
        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.is_feature = is_feature
        self.is_text = is_text
        self.batchnorm_text = nn.BatchNorm1d(emb_dim)
        if is_feature and is_text:
            self.project = nn.Linear(feature_len, emb_dim)
            emb_dim = 2*emb_dim
        elif is_feature:
            self.project = nn.Linear(feature_len, emb_dim)
        self.linear = nn.Linear(emb_dim, 500)
        self.linear2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.LayerNorm(300)
        self.batchnorm_out = nn.BatchNorm1d(500)

    def forward(self, data, length, feature=None):
        """

        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        if self.is_feature and self.is_text:
            out = self.dropout(self.embed(data))
            out = torch.sum(out, dim=1)
            out /= length.float()
            out = self.batchnorm_text(out)
            feature = self.dropout(F.elu(self.batchnorm(self.project(feature))))
            out = torch.cat([out, feature], axis=1)
        elif self.is_feature:
            out = self.dropout(F.elu(self.batchnorm(self.project(feature))))
        else:
            out = self.dropout(self.embed(data))
            out = torch.sum(out, dim=1)
            out /= length.float()
            out = self.batchnorm_text(out)

        # return logits
        out = F.elu(self.batchnorm_out(self.linear(out)))
        out = self.dropout(out)
        out = self.linear2(out)
        return out
