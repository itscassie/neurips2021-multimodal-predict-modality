"""
concrete selection model
this model select import feature from data perspective
aims to gex the feature index that can best reconstruct the original data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteSelect(nn.Module):
    """concrete selection module"""

    def __init__(self, input_dim, select_dim):
        super(ConcreteSelect, self).__init__()
        self.input_dim = input_dim
        self.select_dim = select_dim

        logits_weights = torch.Tensor(select_dim, input_dim)
        self.logits = nn.Parameter(logits_weights)
        nn.init.xavier_normal_(self.logits)

    def forward(self, X, temp):
        selections = F.gumbel_softmax(self.logits, tau=temp, hard=True)
        Y = torch.mm(X, torch.t(selections))

        return Y
