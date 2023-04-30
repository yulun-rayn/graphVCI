import torch

def masked_softmax(logits, mask, dim=1, eps=1e-10):
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    odds = torch.exp(logits-logits_max)

    odds = odds * mask.to(odds.dtype)
    return odds / (torch.sum(odds, dim=dim, keepdim=True) + eps)
