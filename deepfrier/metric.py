import torch

def area_under_roc(pred, target, dim=0):
    """
    Area under receiver operating characteristic curve (ROC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True, dim=dim)
    target = target.gather(dim, order)
    hit = target.cumsum(dim)
    hit = torch.where(target == 0, hit, torch.zeros_like(hit))
    all = (target == 0).sum(dim) * (target == 1).sum(dim)
    auroc = hit.sum(dim) / (all + 1e-10)
    return auroc


def area_under_prc(pred, target, dim=0):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True, dim=dim)
    target = target.gather(dim, order)
    precision = target.cumsum(dim) / torch.ones_like(target).cumsum(dim)
    precision = torch.where(target == 1, precision, torch.zeros_like(precision))
    auprc = precision.sum(dim) / ((target == 1).sum(dim) + 1e-10)
    return auprc


def f1_max(pred, target):
    """
    Protein-centric maximum F1 score.
    First, for each threshold t, we calculate the precision and recall for each protein and
    take their average. Then, we compute the f1 score for each threshold and then take its
    maximum value over all thresholds.

    Parameters:
        pred (Tensor): predictions of shape :math:`(n, d)`
        target (Tensor): binary targets of shape :math:`(n, d)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    all_f1 = all_f1[~torch.isnan(all_f1)]
    return all_f1.max()