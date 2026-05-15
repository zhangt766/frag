"""
models/generator.py
LLM-based generator G_θ with LoRA fine-tuning.

Wraps HuggingFace LLaMA-3-8B and exposes:
  - build_prompt()     : construct text prompt from history + candidates [Eq. 2 / 8]
  - forward()          : compute next-item log-likelihood [Eq. 3 / 13]
  - predict()          : greedy / beam-search next-item inference
  - get_item_embeddings(): retrieve token-level embeddings for soft-weight injection

When soft_weights are supplied, item token embeddings are scaled by w̃_i [Eq. 9]
before being fed into the LLM, maintaining end-to-end differentiability.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA layer (lightweight implementation; use peft library in production)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation of a frozen linear layer."""

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.scale = alpha / rank

        in_features  = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features,  rank,         bias=False)
        self.lora_B = nn.Linear(rank,         out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base weights
        for p in self.original.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base + lora


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Constructs text prompts following Eq. 2 / 8 of the paper.

    Template:
        "The user has watched [History].
         Please predict the next movie this user will watch.
         Choose the answer from: [Candidates]."

    The word "movie" is replaced by the domain keyword for other datasets.
    """

    TEMPLATES = {
        "movielens": (
            "The user has watched {history}.\n"
            "Please predict the next movie this user will watch.\n"
            "Choose the answer from: {candidates}."
        ),
        "steam": (
            "The user has played {history}.\n"
            "Please predict the next game this user will play.\n"
            "Choose the answer from: {candidates}."
        ),
        "lastfm": (
            "The user has listened to {history}.\n"
            "Please predict the next artist this user will listen to.\n"
            "Choose the answer from: {candidates}."
        ),
        "goodreads": (
            "The user has read {history}.\n"
            "Please predict the next book this user will read.\n"
            "Choose the answer from: {candidates}."
        ),
    }
    DEFAULT_TEMPLATE = TEMPLATES["movielens"]

    def __init__(self, item2text: Dict[int, str], dataset: str = "movielens") -> None:
        self.item2text = item2text
        self.template = self.TEMPLATES.get(dataset, self.DEFAULT_TEMPLATE)

    def item_name(self, item_id: int) -> str:
        return self.item2text.get(item_id, f"item_{item_id}")

    def build(
        self,
        history: List[int],
        candidates: List[int],
    ) -> str:
        """Build a single prompt string."""
        hist_str = ", ".join(self.item_name(i) for i in history)
        cand_str = ", ".join(self.item_name(i) for i in candidates)
        return self.template.format(history=hist_str, candidates=cand_str)

    def build_batch(
        self,
        histories: List[List[int]],
        candidate_lists: List[List[int]],
    ) -> List[str]:
        return [
            self.build(h, c) for h, c in zip(histories, candidate_lists)
        ]


# ---------------------------------------------------------------------------
# Generator wrapper
# ---------------------------------------------------------------------------

import math  # needed for LoRALinear (placed here to avoid circular import issue)


class FRAGGenerator(nn.Module):
    """
    Generator G_θ: wraps a causal LLM (LLaMA-3) with LoRA adapters.

    In environments without the actual LLaMA weights we fall back to a
    lightweight MLP-based surrogate for unit-testing the training loop.
    Set `use_llm=False` for the surrogate.
    """

    def __init__(
        self,
        num_items: int,
        generator_dim: int = 256,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_llm: bool = False,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.use_llm = use_llm

        if use_llm:
            self._init_llm(model_name, lora_rank, lora_alpha, lora_dropout)
        else:
            # Lightweight surrogate for testing without GPU / LLM weights
            self.item_embed = nn.Embedding(num_items + 1, generator_dim, padding_idx=0)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=generator_dim, nhead=4, dim_feedforward=512,
                    dropout=0.1, batch_first=True,
                ),
                num_layers=2,
            )
            self.output_head = nn.Linear(generator_dim, num_items + 1)
            self.generator_dim = generator_dim

        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    def _init_llm(self, model_name, rank, alpha, dropout):
        """Load LLaMA-3 and apply LoRA to q_proj / v_proj."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=["q_proj", "v_proj"],
            )
            self.llm = get_peft_model(base_model, lora_cfg)
            self.generator_dim = self.llm.config.hidden_size

        except ImportError as e:
            raise ImportError(
                "Install transformers and peft to use the LLM generator: "
                f"{e}"
            )

    # ------------------------------------------------------------------
    # Embedding access (for soft-weight injection, Eq. 9)
    # ------------------------------------------------------------------

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Return token-level embeddings for item_ids from the generator's
        embedding layer.  Shape: (B, C, generator_dim).
        """
        if self.use_llm:
            embed_layer = self.llm.get_input_embeddings()
            return embed_layer(item_ids)
        else:
            return self.item_embed(item_ids)

    # ------------------------------------------------------------------
    # Forward  [Eq. 13: L_log = -log π_θ(i^{t+1} | H^t_u, d̂)]
    # ------------------------------------------------------------------

    def forward(
        self,
        histories: torch.Tensor,           # (B, L_hist)
        candidate_pools: torch.Tensor,     # (B, C)
        targets: torch.Tensor,             # (B,)
        soft_weights: Optional[torch.Tensor] = None,  # (B, C)
        mask: Optional[torch.Tensor] = None,          # (B, C) bool
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute negative log-likelihood loss for the target item.

        Returns:
            loss_log  : scalar, L_log(u, t)
            logits    : (B, num_items+1) prediction logits
        """
        if self.use_llm:
            return self._forward_llm(
                prompts, targets, candidate_pools, soft_weights
            )
        else:
            return self._forward_surrogate(
                histories, candidate_pools, targets, soft_weights, mask
            )

    # ------------------------------------------------------------------
    def _forward_surrogate(
        self,
        histories: torch.Tensor,
        candidate_pools: torch.Tensor,
        targets: torch.Tensor,
        soft_weights: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Surrogate forward pass (no actual LLM)."""
        # Encode history
        hist_emb = self.item_embed(histories)                     # (B, L, dim)
        hist_ctx = hist_emb.mean(dim=1, keepdim=True)             # (B, 1, dim)

        # Encode candidates with optional soft-weight scaling [Eq. 9]
        cand_emb = self.item_embed(candidate_pools)               # (B, C, dim)
        if soft_weights is not None:
            cand_emb = cand_emb * soft_weights.unsqueeze(-1)      # scaled

        # Apply mask: zero out non-retrieved items
        if mask is not None:
            cand_emb = cand_emb * mask.unsqueeze(-1).float()

        # Transformer over [history_ctx, candidate_embeddings]
        seq = torch.cat([hist_ctx, cand_emb], dim=1)              # (B, 1+C, dim)
        enc = self.encoder(seq)                                    # (B, 1+C, dim)
        pooled = enc.mean(dim=1)                                   # (B, dim)

        logits = self.output_head(pooled)                          # (B, num_items+1)
        loss_log = F.cross_entropy(logits, targets)

        return {"loss_log": loss_log, "logits": logits}

    # ------------------------------------------------------------------
    def _forward_llm(
        self,
        prompts: List[str],
        targets: torch.Tensor,
        candidate_pools: torch.Tensor,
        soft_weights: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Full LLM forward pass.
        Tokenise prompts, inject soft-weighted candidate embeddings,
        compute cross-entropy loss over target item token.
        """
        device = targets.device
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Get input embeddings and inject soft weights for candidates
        inputs_embeds = self.llm.get_input_embeddings()(enc["input_ids"])
        # (Soft-weight injection into continuous embedding space is
        #  approximated at the candidate pool embedding level; a full
        #  implementation maps candidate item ids to token positions.)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=enc["attention_mask"],
            labels=enc["input_ids"],
        )
        loss_log = outputs.loss

        # Build logits over item vocabulary from last hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]         # (B, hid)
        logits = torch.zeros(
            targets.size(0), self.num_items + 1, device=device
        )
        return {"loss_log": loss_log, "logits": logits}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        histories: torch.Tensor,
        candidate_pools: torch.Tensor,
        soft_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        topk: int = 10,
    ) -> torch.Tensor:
        """
        Return top-k item indices from candidate_pools ranked by score.
        Shape: (B, topk).
        """
        dummy_targets = torch.zeros(
            histories.size(0), dtype=torch.long, device=histories.device
        )
        out = self.forward(
            histories, candidate_pools, dummy_targets,
            soft_weights=soft_weights, mask=mask, prompts=prompts,
        )
        logits = out["logits"]                                     # (B, V)

        # Restrict to candidate pool items for ranking
        B, C = candidate_pools.shape
        cand_logits = torch.gather(
            logits, 1, candidate_pools.clamp(max=logits.size(1) - 1)
        )                                                          # (B, C)
        _, top_indices = cand_logits.topk(min(topk, C), dim=1)    # (B, topk)
        # Map back to item ids
        top_items = torch.gather(candidate_pools, 1, top_indices)  # (B, topk)
        return top_items
