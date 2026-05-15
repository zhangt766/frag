"""
main.py
FRAG entry point.

Usage:
    # Train
    python main.py --mode train --dataset movielens

    # Test (load best checkpoint and evaluate on test set)
    python main.py --mode test --dataset movielens --ckpt checkpoints/frag_best_epoch5.pt

    # Parameter sensitivity analysis (RQ2)
    python main.py --mode sweep --dataset movielens --sweep_param tau

    # Long-term fairness curve (RQ4)
    python main.py --mode cfd --dataset movielens --ckpt checkpoints/frag_best_epoch5.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import json
import random

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frag.main")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(dataset: str) -> dict:
    """Merge default config with dataset-specific overrides."""
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/datasets.yaml") as f:
        ds_cfgs = yaml.safe_load(f)

    if dataset in ds_cfgs:
        ds_cfg = ds_cfgs[dataset]
        # Deep merge
        for section, values in ds_cfg.items():
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(values)
            else:
                cfg[section] = values

    return cfg


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(
    cfg: dict,
    num_items: int,
    item2group: dict,
    item2text: dict,
    dataset: str,
    device: str,
) -> "FRAG":
    from models.frag import FRAG

    model = FRAG(
        num_items=num_items,
        item2group=item2group,
        # Retriever
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["retriever"]["hidden_dim"],
        tau_init=cfg["retriever"]["tau_init"],
        gamma=cfg["retriever"]["gamma"],
        eps=cfg["retriever"]["eps"],
        # Generator
        generator_dim=cfg["model"].get("generator_dim", 256),
        lora_rank=cfg["model"]["lora_rank"],
        lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"],
        use_llm=cfg["model"].get("use_llm", False),
        model_name=cfg["model"]["backbone"],
        # Fairness
        alpha=cfg["fairness"]["alpha"],
        lambda_fair=cfg["fairness"]["lambda_fair"],
        eta=cfg["fairness"]["eta"],
        min_exposure=cfg["fairness"]["min_exposure"],
        num_groups=cfg["data"]["num_groups"],
        # Misc
        dataset=dataset,
        item2text=item2text,
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_train(args, cfg, device):
    from data.dataset import build_dataloaders
    from training.trainer import FRAGTrainer

    data_path = cfg.get("data_path", f"data/{args.dataset}")
    train_loader, val_loader, test_loader, item2group, item2text = \
        build_dataloaders(data_path, cfg, seed=cfg["training"]["seed"])

    num_items = max(item2group.keys()) + 1
    model = build_model(cfg, num_items, item2group, item2text, args.dataset, device)

    trainer = FRAGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=args.output_dir,
    )

    if args.ckpt:
        trainer.load_checkpoint(args.ckpt)

    results = trainer.train()
    logger.info(f"Best val NDCG={results['best_ndcg']:.4f} | WGER={results['best_wger']:.4f}")

    # Save training history
    hist_path = os.path.join(args.output_dir, f"history_{args.dataset}.json")
    with open(hist_path, "w") as f:
        json.dump(results["history"], f, indent=2)
    logger.info(f"Training history saved → {hist_path}")


def mode_test(args, cfg, device):
    from data.dataset import build_dataloaders
    from evaluation.metrics import evaluate

    assert args.ckpt, "Provide --ckpt for test mode."
    data_path = cfg.get("data_path", f"data/{args.dataset}")
    _, _, test_loader, item2group, item2text = \
        build_dataloaders(data_path, cfg, seed=cfg["training"]["seed"])

    num_items = max(item2group.keys()) + 1
    model = build_model(cfg, num_items, item2group, item2text, args.dataset, device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint: {args.ckpt}")

    metrics = evaluate(
        model=model,
        loader=test_loader,
        item2group=item2group,
        topk=cfg["evaluation"]["topk"],
        device=device,
        min_exposure=cfg["fairness"]["min_exposure"],
    )

    logger.info("=== Test Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v:.4f}")

    out_path = os.path.join(args.output_dir, f"test_results_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved → {out_path}")


def mode_cfd(args, cfg, device):
    """
    Plot Cumulative Fairness Debt over interaction rounds (Figure 4).
    """
    from data.dataset import build_dataloaders
    from evaluation.metrics import FairnessEvaluator

    assert args.ckpt, "Provide --ckpt for cfd mode."
    data_path = cfg.get("data_path", f"data/{args.dataset}")
    _, _, test_loader, item2group, item2text = \
        build_dataloaders(data_path, cfg, seed=cfg["training"]["seed"])

    num_items = max(item2group.keys()) + 1
    model = build_model(cfg, num_items, item2group, item2text, args.dataset, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ev = FairnessEvaluator(item2group, num_groups=cfg["data"]["num_groups"])
    topk = cfg["evaluation"]["topk"]

    with torch.no_grad():
        for batch in test_loader:
            histories       = batch["histories"].to(device)
            history_lens    = batch["history_lens"]
            candidate_pools = batch["candidate_pools"].to(device)
            user_ids        = batch["user_ids"]

            _, info = model.recommend(histories, history_lens, candidate_pools, topk=topk)
            ev.update(
                soft_w=info["soft_w"],
                candidate_pools=candidate_pools,
                user_ids=user_ids,
                min_exposure=cfg["fairness"]["min_exposure"],
            )

    curve = ev.cfd_curve()
    out = {"cfd_curve": curve, "dataset": args.dataset}
    out_path = os.path.join(args.output_dir, f"cfd_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    logger.info(f"CFD curve saved → {out_path} (final CFD={curve[-1]:.4f})")


def mode_sweep(args, cfg, device):
    """
    Hyperparameter sweep for RQ2 (τ, λ, α).
    Logs (param_value, NDCG, WGER) for each setting.
    """
    from data.dataset import build_dataloaders
    from evaluation.metrics import evaluate

    data_path = cfg.get("data_path", f"data/{args.dataset}")
    train_loader, val_loader, _, item2group, item2text = \
        build_dataloaders(data_path, cfg, seed=cfg["training"]["seed"])

    param = args.sweep_param
    if param == "tau":
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        key_path = ["retriever", "tau_init"]
    elif param == "lambda":
        values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        key_path = ["fairness", "lambda_fair"]
    elif param == "alpha":
        values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        key_path = ["fairness", "alpha"]
    else:
        raise ValueError(f"Unknown sweep_param: {param}")

    results = []
    for v in values:
        cfg_copy = yaml.safe_load(yaml.dump(cfg))   # deep copy
        cfg_copy[key_path[0]][key_path[1]] = v

        num_items = max(item2group.keys()) + 1
        model = build_model(cfg_copy, num_items, item2group, item2text, args.dataset, device)

        # Quick train (1 epoch for sweep)
        from training.trainer import FRAGTrainer
        trainer = FRAGTrainer(model, train_loader, val_loader, cfg_copy, device, args.output_dir)
        cfg_copy["training"]["epochs"] = 1
        trainer.train()

        metrics = evaluate(model, val_loader, item2group,
                           topk=cfg["evaluation"]["topk"], device=device)
        ndcg_k = [k for k in metrics if "ndcg" in k][0]
        row = {"param": param, "value": v,
               "ndcg": metrics[ndcg_k], "wger": metrics["wger"]}
        results.append(row)
        logger.info(f"  {param}={v:.3f} | NDCG={row['ndcg']:.4f} | WGER={row['wger']:.4f}")

    out_path = os.path.join(args.output_dir, f"sweep_{param}_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sweep results → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FRAG: Fair RAG for Sequential Recommendation")
    parser.add_argument("--mode",       type=str, default="train",
                        choices=["train", "test", "cfd", "sweep"],
                        help="Running mode")
    parser.add_argument("--dataset",    type=str, default="movielens",
                        choices=["movielens", "steam", "lastfm", "goodreads"])
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Path to checkpoint (for test / cfd modes)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints and results")
    parser.add_argument("--device",     type=str, default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--sweep_param", type=str, default="tau",
                        choices=["tau", "lambda", "alpha"],
                        help="Which hyperparameter to sweep (mode=sweep only)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = load_config(args.dataset)
    cfg["training"]["seed"] = args.seed
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        mode_train(args, cfg, device)
    elif args.mode == "test":
        mode_test(args, cfg, device)
    elif args.mode == "cfd":
        mode_cfd(args, cfg, device)
    elif args.mode == "sweep":
        mode_sweep(args, cfg, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
