"""
training/trainer.py
FRAG training procedure – implements Algorithm 1 from the paper.

Algorithm 1:
  Input : training tuples {(H^t_u, i^{t+1})}, epochs E, learning rate ρ
  Output: optimised parameters (φ★, θ★, τ★)
  1. Initialise φ, θ, τ, R ← 0
  2. for epoch = 1 … E:
  3.   for each mini-batch B:
  4.     for each (u, t) ∈ B:
  5.       Compute s_{u,i}      [Eq. 5]
  6.       Compute w̃_i, d̃      [Eq. 6]
  7.       Rank items → d̂
  8.       Compute E_g(u,t)     [Eq. 10]
  9.       Compute V(u,t)       [Eq. 12]
  10.      Update R_t           [Eq. 11]
  11.      Form prompt P̂
  12.      Compute L_log, L_reg [Eq. 13, 14]
  13.      L ← L + L_log + η L_reg
  14.    Update φ, θ, τ via gradient descent on L
  15. Return (φ★, θ★, τ★)
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.frag import FRAG
from evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: separate parameter groups for φ, θ, τ
# ---------------------------------------------------------------------------

def build_optimizers(
    model: FRAG,
    lr_retriever: float = 1e-3,
    lr_generator: float = 2e-5,
    lr_tau: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Two optimisers:
      opt_retriever : updates φ (retriever params) + τ (threshold)
      opt_generator : updates θ (generator / LoRA params)

    This mirrors Algorithm 1 line 18: "Update φ, θ, τ by gradient descent on L".
    Using separate optimisers allows different learning rates.
    """
    retriever_params = list(model.retriever.item_encoder.parameters()) + \
                       list(model.retriever.user_encoder.parameters()) + \
                       list(model.retriever.scorer.parameters())

    tau_params = [model.retriever.tau]

    generator_params = [p for p in model.generator.parameters() if p.requires_grad]

    opt_retriever = AdamW(
        [
            {"params": retriever_params, "lr": lr_retriever},
            {"params": tau_params,       "lr": lr_tau},
        ],
        weight_decay=weight_decay,
    )
    opt_generator = AdamW(
        generator_params,
        lr=lr_generator,
        weight_decay=weight_decay,
    )
    return opt_retriever, opt_generator


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FRAGTrainer:
    """
    Orchestrates the full FRAG training loop (Algorithm 1).

    Args:
        model         : FRAG model instance
        train_loader  : training DataLoader
        val_loader    : validation DataLoader
        cfg           : nested config dict (loaded from configs/*.yaml)
        device        : torch device string
        output_dir    : directory to save checkpoints
    """

    def __init__(
        self,
        model: FRAG,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: str = "cuda",
        output_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        train_cfg = cfg["training"]
        self.epochs      = train_cfg["epochs"]
        self.eta         = cfg["fairness"]["eta"]

        # Optimisers
        self.opt_ret, self.opt_gen = build_optimizers(
            model,
            lr_retriever=train_cfg["lr_retriever"],
            lr_generator=train_cfg["lr_generator"],
            lr_tau=train_cfg["lr_tau"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Cosine LR schedulers
        total_steps = self.epochs * len(train_loader)
        self.sched_ret = CosineAnnealingLR(self.opt_ret, T_max=total_steps)
        self.sched_gen = CosineAnnealingLR(self.opt_gen, T_max=total_steps)

        self.best_val_ndcg  = -float("inf")
        self.best_val_wger  = -float("inf")
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """One training epoch – Algorithm 1 inner loops."""
        self.model.train()
        self.model.fairness_tracker.reset()

        total_loss = total_log = total_reg = 0.0
        total_V = total_R = 0.0
        n_batches = 0

        for step, batch in enumerate(self.train_loader):
            histories       = batch["histories"].to(self.device)
            history_lens    = batch["history_lens"]
            candidate_pools = batch["candidate_pools"].to(self.device)
            targets         = batch["targets"].to(self.device)

            # ----- Forward pass (Algorithm 1, lines 8-16) -----
            out = self.model(
                histories=histories,
                history_lens=history_lens,
                candidate_pools=candidate_pools,
                targets=targets,
                update_fairness=True,
            )

            loss = out["loss"]

            # ----- Backward + optimise -----
            self.opt_ret.zero_grad()
            self.opt_gen.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.opt_ret.step()
            self.opt_gen.step()
            self.sched_ret.step()
            self.sched_gen.step()

            total_loss += loss.item()
            total_log  += out["loss_log"].item()
            total_reg  += out["loss_reg"].item()
            total_V    += out["V"]
            total_R    += out["R_t"]
            n_batches  += 1

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / n_batches
                tau_val  = self.model.retriever.tau.item()
                avg_k    = out["mask"].float().sum(dim=1).mean().item()
                logger.info(
                    f"Epoch {epoch} | step {step+1}/{len(self.train_loader)} "
                    f"| loss={avg_loss:.4f} | τ={tau_val:.4f} "
                    f"| avg_K={avg_k:.1f} | R_t={out['R_t']:.4f}"
                )

        return {
            "train_loss":     total_loss  / n_batches,
            "train_loss_log": total_log   / n_batches,
            "train_loss_reg": total_reg   / n_batches,
            "train_V_avg":    total_V     / n_batches,
            "train_R_final":  total_R     / n_batches,
            "tau":            self.model.retriever.tau.item(),
        }

    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Evaluate on validation set."""
        item2group = self.model.item2group
        topk = self.cfg["evaluation"]["topk"]
        min_exp = self.cfg["fairness"]["min_exposure"]

        metrics = evaluate(
            model=self.model,
            loader=self.val_loader,
            item2group=item2group,
            topk=topk,
            device=self.device,
            min_exposure=min_exp,
        )
        return metrics

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, tag: str = "best") -> None:
        path = os.path.join(self.output_dir, f"frag_{tag}_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "opt_ret_state": self.opt_ret.state_dict(),
                "opt_gen_state": self.opt_gen.state_dict(),
                "tau": self.model.retriever.tau.item(),
                "R_t": self.model.fairness_tracker.state,
                "history": self.history,
            },
            path,
        )
        logger.info(f"Saved checkpoint → {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.opt_ret.load_state_dict(ckpt["opt_ret_state"])
        self.opt_gen.load_state_dict(ckpt["opt_gen_state"])
        self.history = ckpt.get("history", [])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    def train(self) -> Dict:
        """
        Full training loop (Algorithm 1, outer loop).
        Returns best validation metrics.
        """
        logger.info(
            f"Starting FRAG training | epochs={self.epochs} "
            f"| device={self.device}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)

            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "elapsed": elapsed,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(record)

            ndcg_key  = [k for k in val_metrics if "ndcg" in k][0]
            wger_key  = "wger"
            cur_ndcg  = val_metrics.get(ndcg_key, 0.0)
            cur_wger  = val_metrics.get(wger_key, 0.0)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | {elapsed:.1f}s | "
                f"NDCG={cur_ndcg:.4f} | WGER={cur_wger:.4f} | "
                f"ED={val_metrics.get('ed', 0):.4f} | "
                f"τ={train_metrics['tau']:.4f}"
            )

            # Save best checkpoint (balance utility + fairness)
            combined = cur_ndcg + cur_wger
            best_combined = self.best_val_ndcg + self.best_val_wger
            if combined > best_combined:
                self.best_val_ndcg = cur_ndcg
                self.best_val_wger = cur_wger
                self._save_checkpoint(epoch, tag="best")

            # Always save latest
            self._save_checkpoint(epoch, tag="latest")

        logger.info("Training complete.")
        return {
            "best_ndcg": self.best_val_ndcg,
            "best_wger": self.best_val_wger,
            "history": self.history,
        }
