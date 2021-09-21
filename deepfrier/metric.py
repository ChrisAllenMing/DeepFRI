import os
import sys

import numpy as np
from numpy.core.numeric import ones

import torch
from torch.nn import functional as F
from torch_scatter import scatter_max
import networkx as nx
from rdkit import Chem
from rdkit.Chem import RDConfig, Descriptors

from torchdrug import utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

# sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# import sascorer


@R.register("metrics.auroc")
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


@R.register("metrics.auprc")
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


@R.register("metrics.f1_max")
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
    return all_f1.max()


@R.register("metrics.r2")
def r2(pred, target):
    """
    :math:`R^2` regression score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): targets of shape :math:`(n,)`
    """
    total = torch.var(target, unbiased=False)
    residual = F.mse_loss(pred, target)
    return 1 - residual / total


@R.register("metrics.logp")
def logP(pred):
    """
    Logarithm of partition coefficient between octanol and water for a compound.

    Parameters:
        pred (data.PackedMolecule): molecules to evaluate
    """
    logp = []
    for mol in pred:
        mol = mol.to_molecule()
        try:
            with utils.no_rdkit_log():
                mol.UpdatePropertyCache()
                score = Descriptors.MolLogP(mol)
        except Chem.AtomValenceException:
            score = 0
        logp.append(score)

    return torch.tensor(logp, dtype=torch.float, device=pred.device)


@R.register("metrics.penalized_logp")
def penalized_logP(pred):
    """
    Logarithm of partition coefficient, penalized by cycle length and synthetic accessibility.

    Parameters:
        pred (data.PackedMolecule): molecules to evaluate
    """
    # statistics from ZINC250k
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = 3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = 0.0485696876403053
    cycle_std = 0.2860212110245455

    plogp = []
    for mol in pred:
        cycles = nx.cycle_basis(nx.Graph(mol.edge_list[:, :2].tolist()))
        if cycles:
            max_cycle = max([len(cycle) for cycle in cycles])
            cycle = max(0, max_cycle - 6)
        else:
            cycle = 0
        mol = mol.to_molecule()
        try:
            with utils.no_rdkit_log():
                mol.UpdatePropertyCache()
                Chem.GetSymmSSSR(mol)
                logp = Descriptors.MolLogP(mol)
                sa = sascorer.calculateScore(mol)
            logp = (logp - logp_mean) / logp_std
            sa = (sa - sa_mean) / sa_std
            cycle = (cycle - cycle_mean) / cycle_std
            score = logp - sa - cycle
        except Chem.AtomValenceException:
            score = -30
        plogp.append(score)

    return torch.tensor(plogp, dtype=torch.float, device=pred.device)


@R.register("metrics.SA")
def SA(pred):
    """
    Synthetic accesibility score.

    Parameters:
        pred (data.PackedMolecule): molecules to evaluate
    """
    sa = []
    for mol in pred:
        with utils.no_rdkit_log():
            score = sascorer.calculateScore(mol.to_molecule())
        sa.append(score)

    return torch.tensor(sa, dtype=torch.float, device=pred.device)


@R.register("metrics.qed")
def QED(pred):
    """
    Quantitative estimation of drug-likeness.

    Parameters:
        pred (data.PackedMolecule): molecules to evaluate
    """
    qed = []
    for mol in pred:
        try:
            with utils.no_rdkit_log():
                score = Descriptors.qed(mol.to_molecule())
        except Chem.AtomValenceException:
            score = -1
        qed.append(score)

    return torch.tensor(qed, dtype=torch.float, device=pred.device)


@R.register("metrics.validity")
def chemical_validity(pred):
    """
    Chemical validity of molecules.

    Parameters:
        pred (data.PackedMolecule): molecules to evaluate
    """
    validity = []
    for i, mol in enumerate(pred):
        with utils.no_rdkit_log():
            smiles = mol.to_smiles()
            mol = Chem.MolFromSmiles(smiles)
        validity.append(1 if mol else 0)

    return torch.tensor(validity, dtype=torch.float, device=pred.device)


def variadic_accuracy(input, target, size):
    """
    Compute classification accuracy over variadic sizes of categories.

    Suppose there are :math:`N` samples, and the number of categories in all samples is summed to :math`B`.

    Parameters:
        input (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(N,)`. Each target is a relative index in a sample.
        size (Tensor): number of categories of shape :math:`(N,)`
    """
    index2graph = functional._size_to_index(size)

    input_class = scatter_max(input, index2graph)[1]
    target_index = target + size.cumsum(0) - size
    accuracy = (input_class == target_index).float()
    return accuracy


@R.register("metrics.mse")
def mean_squared_error(pred, target) -> float:
    """
    Mean square error between target and prediction.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    return torch.mean((pred - target).pow(2))


@R.register("metrics.mae")
def mean_absolute_error(pred, target) -> float:
    """
    Mean absolute error between target and prediction.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    return torch.mean((pred - target).abs())


@R.register("metrics.spearmanr")
def spearmanr(pred, target) -> float:
    """
    Spearman correlation between target and prediction. 
    Implement in PyTorch, but non-diffierentiable. (validation metric only)

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    # return ascending ranking indices (from 0, smallest) of tensor
    def _get_ranks(x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(len(x)).to(x.device)
        return ranks
    pred = pred.view(-1)
    target = target.view(-1)
    pred_rank = _get_ranks(pred)
    target_rank = _get_ranks(target)
    n = pred.size(0)    # length
    numerator = 6.0 * torch.sum((pred_rank - target_rank).pow(2))
    denominator = n * (n ** 2 - 1.0)
    _spearmanr = 1.0 - (numerator / denominator)
    return _spearmanr


@R.register("metrics.accuracy")
def accuracy(pred, target, ignore_index=-1) -> float:
    """
    Calculate accuracy for classification task. Support sequence target.
    Implemented following TAPE benchmark.
    See 'https://github.com/songlab-cal/tape/blob/master/tape/metrics.py'
    
    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
        ignore_index (int): ignore corresponding elements in both array 
            when calculate accuracy. (from TAPE)
    """
    if isinstance(target[0], int):  # non-sequence target label, eg. [3] as label 
        return torch.mean((pred.argmax(-1) == target).float())
    else:                           # sequence target 
        correct, total = 0., 0.
        pred_labels = pred.argmax(-1)
        for label_array, pred_array in zip(target, pred_labels):
            # ignore specific label
            mask = label_array != ignore_index
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size(0)
        return correct / total


@R.register("metrics.mcc")
def matthews_corr_coef(pred, target, epsilon=1e-6) -> float:
    """
    Matthews correlation coefficient between target and prediction.
    
    Definition follows matthews_corrcoef for K classes in sklearn.
    For details, see: 'https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef'

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    nb_classes = pred.size(-1)
    pred_labels = pred.argmax(-1)
    confusion_matrix = torch.zeros(nb_classes, nb_classes).to(pred.device)
    for t, p in zip(target.view(-1), pred_labels.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    tk = torch.sum(confusion_matrix, dim=0)
    pk = torch.sum(confusion_matrix, dim=1)
    c = torch.sum(confusion_matrix.diag())
    s = torch.sum(confusion_matrix)
    mcc = (c * s - torch.sum(tk*pk)) / \
          (torch.sqrt((s*s - torch.sum(pk.pow(2))) * (s*s - torch.sum(tk.pow(2)))) + epsilon)
    return mcc
    