"""
models/frag.py
FRAG: Fair Retrieval-Augmented Generation for Sequential Recommendation.

Integrates:
  - AdaptiveRetriever  (models/retriever.py)
  - FRAGGenerator      (models/generator.py)
  - Online fairness-risk state R_t  [Eq. 10-12]
  - Joint training objective L_train [Eq. 15]

Algorithm 1 from the paper is implemented in training/trainer.py;
this module owns the model logic and loss computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.retriever import AdaptiveRetriever
from models.generator import FRAGGenerator, PromptBuilder


# ---------------------------------------------------------------------------
# Online Fairness-Risk State (EMA tracker)
# ---------------------------------------------------------------------------

class OnlineFairnessRisk:
    """
    Maintains the EMA fairness-risk state R_t  [Eq. 11].

    R_t = α R_{t-1} + (1 − α) V(u, t)
    V(u, t) = max_{g ∈ G} (m_g − E_g(u, t))_+    [Eq. 12]

    Attributes:
        R        : current EMA state (scalar, on CPU)
        alpha    : EMA smoothing coefficient
        min_exp  : minimum desired exposure m_g (uniform across groups)
        num_groups: number of item/provider groups
    """

    def __init__(
        self,
        alpha: float = 0.9,
        min_exposure: float = 0.1,
        num_groups: int = 2,
    ) -> None:
        self.alpha = alpha
        self.min_exposure = min_exposure
        self.num_groups = num_groups
        self.R: float = 0.0

    def reset(self) -> None:
        self.R = 0.0

    def compute_group_exposure(
        self,
        soft_w: torch.Tensor,          # (B, C) soft weights
        candidate_pools: torch.Tensor, # (B, C) item ids
        item2group: Dict[int, int],
    ) -> torch.Tensor:                 # (num_groups,) E_g(u,t) summed over batch
        """
        E_g(u, t) = Σ_{i ∈ G_g} w̃_i(H^t_u; φ, τ)   [Eq. 10]

        Averaged over the batch dimension.
        """
        B, C = soft_w.shape
        device = soft_w.device

        # Build group mask: (B, C, num_groups)
        group_ids = torch.zeros(B, C, dtype=torch.long, device=device)
        for b in range(B):
            for c in range(C):
                item_id = candidate_pools[b, c].item()
                group_ids[b, c] = item2group.get(item_id, 0)

        group_mask = F.one_hot(group_ids, self.num_groups).float()  # (B,C,G)
        # Weighted sum per group per sample: (B, G)
        exposure = torch.einsum("bc,bcg->bg", soft_w, group_mask)
        # Average over batch
        return exposure.mean(dim=0)                                  # (G,)

    def instantaneous_risk(
        self,
        group_exposure: torch.Tensor,   # (num_groups,)
    ) -> torch.Tensor:                  # scalar
        """
        V(u, t) = max_{g} (m_g − E_g)_+   [Eq. 12]
        """
        shortfall = (self.min_exposure - group_exposure).clamp(min=0)
        return shortfall.max()

    def update(self, V: float) -> float:
        """
        R_t = α R_{t-1} + (1 − α) V   [Eq. 11]
        Returns the updated R_t.
        """
        self.R = self.alpha * self.R + (1 - self.alpha) * V
        return self.R

    @property
    def state(self) -> float:
        return self.R


# ---------------------------------------------------------------------------
# FRAG Main Model
# ---------------------------------------------------------------------------

class FRAG(nn.Module):
    """
    FRAG: Fair Retrieval-Augmented Generation.

    Joint model that owns φ (retriever), θ (generator), τ (threshold),
    and the online fairness-risk state R_t.

    Loss  [Eq. 15]:
        L_train(φ, θ, τ) = Σ_{u,t} [L_log(u,t) + η L_reg(u,t)]

    where
        L_log(u, t) = -log π_θ(i^{t+1} | H^t_u, d̂)      [Eq. 13]
        L_reg(u, t) = Σ_i w̃_i + λ R_t                    [Eq. 14]
    """

    def __init__(
        self,
        num_items: int,
        item2group: Dict[int, int],
        # Retriever config
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        tau_init: float = 0.3,
        gamma: float = 0.1,
        eps: float = 1e-6,
        # Generator config
        generator_dim: int = 256,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_llm: bool = False,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        # Fairness config
        alpha: float = 0.9,
        lambda_fair: float = 1.0,
        eta: float = 1.0,
        min_exposure: float = 0.1,
        num_groups: int = 2,
        # Misc
        dataset: str = "movielens",
        item2text: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__()

        self.retriever = AdaptiveRetriever(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            tau_init=tau_init,
            gamma=gamma,
            eps=eps,
        )
        self.generator = FRAGGenerator(
            num_items=num_items,
            generator_dim=generator_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_llm=use_llm,
            model_name=model_name,
        )

        self.item2group = item2group
        self.fairness_tracker = OnlineFairnessRisk(
            alpha=alpha,
            min_exposure=min_exposure,
            num_groups=num_groups,
        )

        self.lambda_fair = lambda_fair
        self.eta = eta

        # Prompt builder (for LLM mode)
        self.prompt_builder = PromptBuilder(
            item2text=item2text or {}, dataset=dataset
        )

    # ------------------------------------------------------------------
    # Group exposure + fairness regularisation
    # ------------------------------------------------------------------

    def compute_L_reg(
        self,
        soft_w: torch.Tensor,          # (B, C)
        candidate_pools: torch.Tensor, # (B, C)
        update_tracker: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """
        L_reg(u, t) = Σ_i w̃_i  +  λ R_t    [Eq. 14]

        Args:
            update_tracker : if True, update the EMA state R_t in-place.

        Returns:
            L_reg  : differentiable scalar tensor
            V_val  : instantaneous fairness risk (float, for logging)
        """
        # Term 1: penalise long candidate sets
        length_penalty = soft_w.sum(dim=1).mean()               # scalar

        # Term 2: fairness risk
        group_exp = self.fairness_tracker.compute_group_exposure(
            soft_w, candidate_pools, self.item2group
        )
        V = self.fairness_tracker.instantaneous_risk(group_exp)
        V_val = V.item()

        if update_tracker:
            self.fairness_tracker.update(V_val)

        R_t = self.fairness_tracker.state
        fairness_penalty = torch.tensor(
            self.lambda_fair * R_t,
            dtype=soft_w.dtype,
            device=soft_w.device,
            requires_grad=False,
        )

        L_reg = length_penalty + fairness_penalty
        return L_reg, V_val

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        histories: torch.Tensor,         # (B, L_hist)
        history_lens: torch.Tensor,      # (B,)
        candidate_pools: torch.Tensor,   # (B, C)
        targets: torch.Tensor,           # (B,)
        prompts: Optional[List[str]] = None,
        update_fairness: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full FRAG forward pass (Algorithm 1, lines 8-16).

        Returns:
            loss       : total training loss L_train for this batch
            loss_log   : recommendation log-likelihood loss
            loss_reg   : fairness regularisation loss
            soft_w     : soft weights (B, C)
            mask       : adaptive set mask (B, C)
            V          : instantaneous fairness risk (float)
            R_t        : EMA fairness state (float)
        """
        # --- Step 1: Retriever ---
        history_lens_t = torch.tensor(
            history_lens, dtype=torch.long, device=histories.device
        ) if not isinstance(history_lens, torch.Tensor) else history_lens

        ret_out = self.retriever(histories, history_lens_t, candidate_pools)
        scores  = ret_out["scores"]   # (B, C)
        soft_w  = ret_out["soft_w"]   # (B, C)
        mask    = ret_out["mask"]     # (B, C) bool

        # --- Step 2: Soft-weight injection into generator embedding space ---
        # Generator embeddings are scaled by soft weights [Eq. 9]
        # (In LLM mode this is done inside _forward_llm; here we pass soft_w)

        # --- Step 3: Generator ---
        gen_out = self.generator(
            histories=histories,
            candidate_pools=candidate_pools,
            targets=targets,
            soft_weights=soft_w,
            mask=mask,
            prompts=prompts,
        )
        loss_log = gen_out["loss_log"]

        # --- Step 4: Fairness regularisation ---
        loss_reg, V_val = self.compute_L_reg(
            soft_w, candidate_pools, update_tracker=update_fairness
        )

        # --- Step 5: Joint loss [Eq. 15] ---
        loss = loss_log + self.eta * loss_reg

        return {
            "loss":     loss,
            "loss_log": loss_log.detach(),
            "loss_reg": loss_reg.detach(),
            "soft_w":   soft_w.detach(),
            "mask":     mask,
            "V":        V_val,
            "R_t":      self.fairness_tracker.state,
            "logits":   gen_out["logits"].detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self,
        histories: torch.Tensor,
        history_lens: torch.Tensor,
        candidate_pools: torch.Tensor,
        prompts: Optional[List[str]] = None,
        topk: int = 10,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Inference: retrieve candidates then rank with generator.

        Returns:
            top_items : (B, topk) recommended item ids
            info      : dict with soft_w, mask, avg_retrieved_k
        """
        history_lens_t = torch.tensor(
            history_lens, dtype=torch.long, device=histories.device
        ) if not isinstance(history_lens, torch.Tensor) else history_lens

        ret_out = self.retriever(histories, history_lens_t, candidate_pools)
        soft_w  = ret_out["soft_w"]
        mask    = ret_out["mask"]

        avg_k = mask.float().sum(dim=1).mean().item()

        top_items = self.generator.predict(
            histories=histories,
            candidate_pools=candidate_pools,
            soft_weights=soft_w,
            mask=mask,
            prompts=prompts,
            topk=topk,
        )

        return top_items, {
            "soft_w": soft_w,
            "mask": mask,
            "avg_retrieved_k": avg_k,
        }
