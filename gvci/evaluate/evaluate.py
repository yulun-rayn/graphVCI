from sklearn.metrics import roc_auc_score

import torch

@torch.no_grad()
def evaluate_auc(data, model):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
