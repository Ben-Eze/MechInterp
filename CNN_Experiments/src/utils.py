import torch


def accuracy_function(pred, truth):
    return torch.sum(torch.eq(pred, truth)) / torch.numel(truth)