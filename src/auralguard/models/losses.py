"""Training objectives for AuralGuard.

* OCSoftmax     — one-class softmax (Zhang et al., 2021). Learns a single target
                  direction for bona-fide and pushes spoofs away; strong for open-set
                  generalization to unseen attacks.
* SupConLoss    — supervised contrastive loss (Khosla et al., 2020), auxiliary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmax(nn.Module):
    """One-class softmax head.

    Parameters
    ----------
    feat_dim : embedding dimension.
    m_real   : angular margin for the bona-fide (target) class.
    m_fake   : angular margin for the spoof class.
    alpha    : scale (inverse temperature).

    Convention: label 0 = bona-fide (target), label 1 = spoof.
    Returns (loss, score) where a HIGHER score = more spoof-like.
    """

    def __init__(self, feat_dim: int = 160, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        nn.init.kaiming_uniform_(self.center, a=5**0.5)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        w = F.normalize(self.center, dim=1)
        x = F.normalize(x, dim=1)
        cos = x @ w.t()  # (B, 1) cosine similarity to the bona-fide center
        cos = cos.squeeze(1)

        # spoof score: distance from the bona-fide direction
        score = -cos  # higher when far from bona-fide => more spoof-like

        if labels is None:
            return None, score

        # margins: bona-fide (0) pulled toward center; spoof (1) pushed away
        m = torch.where(labels == 0, torch.full_like(cos, self.m_real),
                        torch.full_like(cos, self.m_fake))
        # sign: target wants cos>m_real; spoof wants cos<m_fake
        sign = torch.where(labels == 0, torch.ones_like(cos), -torch.ones_like(cos))
        logits = self.alpha * sign * (m - cos)
        loss = F.softplus(logits).mean()
        return loss, score


class SupConLoss(nn.Module):
    """Supervised contrastive loss on L2-normalized projections.

    Parameters
    ----------
    temperature : float
        Contrastive temperature.
    bona_fide_only : bool
        If True, only bona-fide (label=0) samples participate as anchors and positives.
        This avoids conflicting with OCSoftmax: SupCon normally tries to cluster spoof
        embeddings together, but OCSoftmax wants spoofs spread freely outside the
        bona-fide manifold. Setting bona_fide_only=True tightens the bona-fide cluster
        without constraining spoof placement.
    """

    def __init__(self, temperature: float = 0.1, bona_fide_only: bool = True):
        super().__init__()
        self.t = temperature
        self.bona_fide_only = bona_fide_only

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=1)

        if self.bona_fide_only:
            # Keep only bona-fide (label=0) samples
            mask_bf = labels == 0
            if mask_bf.sum() < 2:
                return feats.new_zeros(())  # need at least 2 bona-fide samples
            feats = feats[mask_bf]
            labels = labels[mask_bf]

        sim = feats @ feats.t() / self.t
        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim)

        B = feats.size(0)
        eye = torch.eye(B, device=feats.device, dtype=torch.bool)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye

        denom = exp_sim.masked_fill(eye, 0).sum(dim=1, keepdim=True) + 1e-12
        log_prob = sim - torch.log(denom)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return feats.new_zeros(())
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]
        return -mean_log_prob_pos.mean()


def layer_attention_entropy_reg(weights: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the SSL layer-attention weights (encourages spread)."""
    p = F.softmax(weights, dim=0)
    ent = -(p * torch.log(p + 1e-12)).sum()
    return -ent  # minimizing this maximizes entropy
