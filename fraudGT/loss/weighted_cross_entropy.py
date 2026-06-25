import torch
import torch.nn.functional as F
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_loss


@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(pred, true, epoch):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        known_mask = true >= 0
        if not torch.all(known_mask):
            pred = pred[known_mask]
            true = true[known_mask]
        if true.numel() == 0:
            return pred.sum() * 0, pred

        # calculating label weights for weighted loss computation
        if cfg.model.loss_fun_weight is None:
            V = true.size(0)
            n_classes = pred.shape[1] if pred.ndim > 1 else 2
            label_count = torch.bincount(true)
            label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
            cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
            cluster_sizes[torch.unique(true)] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes > 0).float()
        else:
            weight = torch.tensor(cfg.model.loss_fun_weight, device=torch.device(cfg.device))
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)
