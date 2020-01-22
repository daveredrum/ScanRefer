import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRankingLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=1)

        # reduction
        loss = -torch.sum(torch.log(1 - probs + 1e-8) * (1 - targets) * self.weights[0] + torch.log(probs + 1e-8) * targets * self.weights[1], dim=1).mean()

        return loss