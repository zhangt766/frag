"""
evaluation/metrics.py
Utility and long-term fairness evaluation metrics for FRAG.

Utility metrics  (Section 5.2):
  - Recall@K
  - Precision@K
  - MRR@K
  - NDCG@K

Long-term fairness metrics  (Section 5.2):
  - ED   : Exposure Deviation
  - WGER : Worst-Group Exposure Ratio
  - GC   : Group Coverage
  - CFD  : Cumulative Fairness Debt (Section 5.6)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    predicted: List[List[int]],
    targets: List[int],
    k: int = 10,
) -> float:
    """Recall@K: fraction of targets found in top-K predictions."""
    hits = 0
    for pred, tgt in zip(predicted, targets):
        if tgt in pred[:k]:
            hits += 1
    return hits / max(len(targets), 1)


def precision_at_k(
    predicted: List[List[int]],
    targets: List[int],
    k: int = 10,
) -> float:
    """Precision@K: fraction of top-K predictions that are the target."""
    total = 0.0
    for pred, tgt in zip(predicted, targets):
        top_k = pred[:k]
        total += (1.0 / k) if tgt in top_k else 0.0
    return total / max(len(targets), 1)


def mrr_at_k(
    predicted: List[List[int]],
    targets: List[int],
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank@K."""
    total = 0.0
    for pred, tgt in zip(predicted, targets):
        top_k = pred[:k]
        if tgt in top_k:
            rank = top_k.index(tgt) + 1
            total += 1.0 / rank
    return total / max(len(targets), 1)


def ndcg_at_k(
    predicted: List[List[int]],
    targets: List[int],
    k: int = 10,
) -> float:
    """Normalised Discounted Cumulative Gain@K."""
    total = 0.0
    ideal = 1.0 / math.log2(2)          # IDCG with target at rank 1
    for pred, tgt in zip(predicted, targets):
        top_k = pred[:k]
        if tgt in top_k:
            rank = top_k.index(tgt) + 1
            total += (1.0 / math.log2(rank + 1)) / ideal
    return total / max(len(targets), 1)


def compute_utility_metrics(
    predicted: List[List[int]],
    targets: List[int],
    k: int = 10,
) -> Dict[str, float]:
    """Compute all utility metrics in one call."""
    return {
        f"recall@{k}":    recall_at_k(predicted, targets, k),
        f"precision@{k}": precision_at_k(predicted, targets, k),
        f"mrr@{k}":       mrr_at_k(predicted, targets, k),
        f"ndcg@{k}":      ndcg_at_k(predicted, targets, k),
    }


# ---------------------------------------------------------------------------
# Long-term fairness metrics
# ---------------------------------------------------------------------------

class FairnessEvaluator:
    """
    Accumulates per-step group exposure across the evaluation horizon
    and computes ED, WGER, GC, and CFD.

    Usage:
        ev = FairnessEvaluator(item2group, num_groups=2)
        for batch in loader:
            ev.update(soft_w, candidate_pools, user_ids)
        results = ev.compute()
    """

    def __init__(
        self,
        item2group: Dict[int, int],
        num_groups: int = 2,
        target_share: Optional[List[float]] = None,  # q_g; uniform if None
    ) -> None:
        self.item2group = item2group
        self.num_groups = num_groups
        self.target_share = (
            target_share
            if target_share is not None
            else [1.0 / num_groups] * num_groups
        )
        # X_g: cumulative exposure per group  [Eq. 5.2]
        self.X: List[float] = [0.0] * num_groups
        # Group coverage: has any user ever retrieved an item from group g?
        self.covered: List[set] = [set() for _ in range(num_groups)]
        # CFD tracking
        self.cfd_steps: List[float] = []
        self.V_history: List[float] = []

    def reset(self) -> None:
        self.X = [0.0] * self.num_groups
        self.covered = [set() for _ in range(self.num_groups)]
        self.cfd_steps = []
        self.V_history = []

    def update(
        self,
        soft_w: torch.Tensor,           # (B, C) soft weights (or hard 0/1)
        candidate_pools: torch.Tensor,  # (B, C) item ids
        user_ids: List[int],
        min_exposure: float = 0.1,
    ) -> None:
        """
        Accumulate group exposure for one batch of interaction steps.
        """
        B, C = soft_w.shape
        sw = soft_w.detach().cpu().numpy()
        pools = candidate_pools.detach().cpu().numpy()

        for b in range(B):
            uid = user_ids[b]
            group_exp = np.zeros(self.num_groups)
            for c in range(C):
                item_id = int(pools[b, c])
                g = self.item2group.get(item_id, 0)
                w = float(sw[b, c])
                group_exp[g] += w
                if w > 0:
                    self.covered[g].add(uid)

            for g in range(self.num_groups):
                self.X[g] += group_exp[g]

            # Instantaneous fairness risk V for CFD
            shortfall = max((min_exposure - group_exp[g]) for g in range(self.num_groups))
            V = max(shortfall, 0.0)
            self.V_history.append(V)

        # CFD at this step = sum of V so far
        self.cfd_steps.append(sum(self.V_history))

    def compute(self) -> Dict[str, float]:
        """Compute ED, WGER, GC, mean CFD."""
        total_X = sum(self.X) + 1e-12
        p = [x / total_X for x in self.X]
        q = self.target_share

        # Exposure Deviation  [Eq. ED]
        ed = sum(abs(p[g] - q[g]) for g in range(self.num_groups)) / self.num_groups

        # Worst-Group Exposure Ratio  [Eq. WGER]
        wger = min(
            (p[g] / max(q[g], 1e-12)) for g in range(self.num_groups)
        )

        # Group Coverage  [Eq. GC]
        num_covered = sum(1 for g in range(self.num_groups) if len(self.covered[g]) > 0)
        gc = num_covered / self.num_groups

        # Cumulative Fairness Debt (last value)
        cfd_final = self.cfd_steps[-1] if self.cfd_steps else 0.0

        return {
            "ed":         ed,
            "wger":       wger,
            "gc":         gc,
            "cfd_final":  cfd_final,
        }

    def cfd_curve(self) -> List[float]:
        """Return the CFD trajectory for plotting (Figure 4 in paper)."""
        return self.cfd_steps


# ---------------------------------------------------------------------------
# Combined evaluator
# ---------------------------------------------------------------------------

def evaluate(
    model,
    loader,
    item2group: Dict[int, int],
    num_groups: int = 2,
    topk: int = 10,
    device: str = "cpu",
    min_exposure: float = 0.1,
) -> Dict[str, float]:
    """
    Run full evaluation over a DataLoader.

    Returns:
        dict with all utility and fairness metrics.
    """
    model.eval()
    fairness_ev = FairnessEvaluator(item2group, num_groups=num_groups)

    all_predicted: List[List[int]] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in loader:
            histories       = batch["histories"].to(device)
            history_lens    = batch["history_lens"]
            candidate_pools = batch["candidate_pools"].to(device)
            targets         = batch["targets"].to(device)
            user_ids        = batch["user_ids"]

            top_items, info = model.recommend(
                histories, history_lens, candidate_pools, topk=topk
            )

            # Collect predictions
            for b in range(top_items.size(0)):
                all_predicted.append(top_items[b].cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            # Update fairness evaluator
            fairness_ev.update(
                soft_w=info["soft_w"],
                candidate_pools=candidate_pools,
                user_ids=user_ids,
                min_exposure=min_exposure,
            )

    utility = compute_utility_metrics(all_predicted, all_targets, k=topk)
    fairness = fairness_ev.compute()

    return {**utility, **fairness}
