import csv
import json
import math
import os
import random
import pickle
from typing import Dict, Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import utils
from diffusion import policies_hybrid


class IdentityDiffusionModel:
    """Fallback sampler that returns the input unchanged."""

    def reverse_samples(self, B, x_in, cond, **kwargs):
        return x_in, [x_in]


def _build_model_from_config(cfg_path, sample_shape, device):
    import models  # lazy import to avoid circular dependency

    cfg_ckpt = OmegaConf.load(cfg_path)
    family = getattr(cfg_ckpt, "family", "cond_diffusion")
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel,
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
        "no_model": models.NoModel,
    }
    if family not in model_types or "model" not in cfg_ckpt:
        return None

    model_cfg = cfg_ckpt.model
    if not OmegaConf.is_config(model_cfg):
        model_cfg = OmegaConf.create(model_cfg)
    # Ensure nested backbone_params is a DictConfig for open_dict compatibility
    if "backbone_params" in model_cfg and not OmegaConf.is_config(model_cfg.backbone_params):
        model_cfg.backbone_params = OmegaConf.create(model_cfg.backbone_params)

    with open_dict(model_cfg):
        model_cfg["input_shape"] = tuple(sample_shape)
        model_cfg["device"] = str(device)
        if "num_classes" in cfg_ckpt:
            model_cfg["num_classes"] = cfg_ckpt.num_classes
    # Pass DictConfig directly so backbone_params remains an OmegaConf node
    model = model_types[family](**model_cfg).to(device)
    return model


def load_model(cfg: Dict[str, Any], device, sample_shape=None):
    if cfg["sa_mode"] == "sa_only":
        return None
    model_path = cfg["diffusion"].get("model_path")
    if not model_path:
        return IdentityDiffusionModel()

    obj = torch.load(model_path, map_location=device)
    if hasattr(obj, "reverse_samples"):
        return obj
    if isinstance(obj, dict):
        for candidate in ["model", "ema", "ema_model"]:
            if candidate in obj and hasattr(obj[candidate], "reverse_samples"):
                return obj[candidate]
        # state-dict style: rebuild model from config near checkpoint
        ckpt_dir = os.path.dirname(model_path)
        cfg_path = None
        for cand in ["config_resolved.yaml", "config.yaml"]:
            cand_full = os.path.join(ckpt_dir, cand)
            if os.path.exists(cand_full):
                cfg_path = cand_full
                break
        if cfg_path and sample_shape is not None:
            model = _build_model_from_config(cfg_path, sample_shape, device)
            if model is not None:
                state_dict = obj.get("model", obj)
                try:
                    model.load_state_dict(state_dict)
                    return model
                except Exception as e:
                    print(f"Failed to load state dict from {model_path}: {e}")
    print(f"Warning: unsupported checkpoint format at {model_path}; using identity sampler.")
    return IdentityDiffusionModel()


def initial_placement(x, random_init: bool):
    if not random_init:
        return x.clone()
    x_rand = x.clone()
    x_rand[..., :2] = 2 * torch.rand_like(x_rand[..., :2]) - 1
    return x_rand


def macro_indices(cond):
    if hasattr(cond, "is_macros"):
        mask = cond.is_macros
        if hasattr(mask, "bool"):
            mask = mask.bool()
        idx = torch.nonzero(mask).flatten()
        if len(idx) > 0:
            return idx
    return None


def propose_swap(x_cur, macro_idx):
    if macro_idx is None or len(macro_idx) < 2:
        return None
    i, j = random.sample(macro_idx.tolist(), 2)
    x_new = x_cur.clone()
    x_new[:, i, :], x_new[:, j, :] = x_cur[:, j, :].clone(), x_cur[:, i, :].clone()
    return x_new


def propose_jitter(x_cur, macro_idx, std):
    if macro_idx is not None and len(macro_idx) > 0:
        idx = random.choice(macro_idx.tolist())
    else:
        idx = random.randrange(x_cur.shape[1])
    x_new = x_cur.clone()
    jitter = std * torch.randn_like(x_new[:, idx, :2])
    x_new[:, idx, :2] = torch.clamp(x_new[:, idx, :2] + jitter, -1.0, 1.0)
    return x_new


def evaluate_cost(x_candidate, x_ref, cond, weights):
    x_cpu = x_candidate.detach().cpu().squeeze(0)
    cond_cpu = cond
    hpwl_norm, hpwl_rescaled = utils.hpwl_fast(x_cpu, cond_cpu, normalized_hpwl=False)
    legality = utils.check_legality_new(x_cpu, x_ref.cpu().squeeze(0), cond_cpu, cond_cpu.is_ports, score=True)
    if hasattr(cond_cpu, "is_macros"):
        macro_mask = (~cond_cpu.is_macros) | cond_cpu.is_ports
        macro_legality = utils.check_legality_new(x_cpu, x_ref.cpu().squeeze(0), cond_cpu, macro_mask, score=True)
    else:
        macro_legality = 1.0
    congestion = 0.0
    cost = (
        weights["hpwl"] * hpwl_rescaled
        + weights["legality"] * (1 - legality)
        + weights["macro_legality"] * (1 - macro_legality)
        + weights["congestion"] * congestion
    )
    metrics = {
        "cost": float(cost),
        "hpwl_rescaled": float(hpwl_rescaled),
        "hpwl_normalized": float(hpwl_norm),
        "legality": float(legality),
        "macro_legality": float(macro_legality),
        "congestion": float(congestion),
    }
    return cost, metrics


def maybe_diffusion_move(model, x_cur, cond, cfg_diffusion):
    if model is None:
        return None
    samples, intermediates = policies_hybrid.diffusion_move(
        model,
        x_cur,
        cond,
        mode=cfg_diffusion.get("mode", "cond"),
        x_init=x_cur,
        noise_level=cfg_diffusion.get("noise_level", 0.0),
        num_steps=cfg_diffusion.get("k_steps", None),
        intermediate_every=cfg_diffusion.get("intermediate_every", 0),
    )
    return samples, intermediates


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


@hydra.main(version_base=None, config_path="configs", config_name="config_sa_hybrid")
def main(cfg: DictConfig):
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    random.seed(cfg_dict["seed"])
    np.random.seed(cfg_dict["seed"])
    torch.manual_seed(cfg_dict["seed"])

    train_set, val_set = utils.load_graph_data_with_config(
        cfg_dict["task"],
        train_data_limit=cfg_dict["train_data_limit"],
        val_data_limit=cfg_dict["val_data_limit"],
    )
    dataset = val_set if len(val_set) > 0 else train_set
    if len(dataset) == 0:
        raise RuntimeError("No data samples found for the requested task.")

    sample_shape = dataset[0][0].shape
    device = torch.device("cpu")
    model = load_model(cfg_dict, device, sample_shape=sample_shape)

    log_dir = os.path.join(cfg_dict["log_dir"], f"{cfg_dict['task']}.sa_hybrid.{cfg_dict['seed']}")
    samples_dir = os.path.join(log_dir, "samples")
    ensure_dir(samples_dir)
    metrics_path = os.path.join(log_dir, "metrics.csv")
    OmegaConf.save(config=cfg, f=os.path.join(log_dir, "config_resolved.yaml"))

    metrics_rows = []
    summaries = []
    move_probs = cfg_dict.get("move_probs", {})
    swap_prob = move_probs.get("swap", 0.5)
    jitter_std = cfg_dict.get("jitter_std", 0.02)

    for sample_idx, (x, cond) in enumerate(dataset):
        cond = cond.to(device)
        x_ref = x.unsqueeze(0) if x.dim() == 2 else x
        x_cur = initial_placement(x_ref, cfg_dict.get("random_init", False))
        best_x = x_cur.clone()

        _, best_metrics = evaluate_cost(best_x, x_ref, cond, cfg_dict["weights"])
        best_cost = best_metrics["cost"]
        cur_cost = best_cost

        macro_idx = macro_indices(cond)

        if cfg_dict["sa_mode"] == "diffusion_only":
            move = maybe_diffusion_move(model, x_cur, cond, cfg_dict["diffusion"])
            if move is not None:
                x_new, _ = move
                new_cost, new_metrics = evaluate_cost(x_new, x_ref, cond, cfg_dict["weights"])
                best_flag = new_cost < best_cost
                if best_flag:
                    best_cost = new_cost
                    best_metrics = new_metrics
                    best_x = x_new.detach().clone()
                metrics_rows.append(
                    {
                        "sample_id": getattr(cond, "file_idx", sample_idx),
                        "step": 0,
                        "move": "diffusion",
                        "temp": 0.0,
                        "accepted": True,
                        **new_metrics,
                    }
                )
            sample_id = getattr(cond, "file_idx", sample_idx)
            best_path = os.path.join(samples_dir, f"best{sample_id}.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(best_x.squeeze(0).cpu(), f)
            print(f"Sample {sample_id}: best cost {best_cost:.4f}")
            summaries.append(
                {
                    "sample_id": getattr(cond, "file_idx", sample_idx),
                    "best_cost": float(best_cost),
                    "hpwl_rescaled": best_metrics["hpwl_rescaled"],
                    "legality": best_metrics["legality"],
                    "macro_legality": best_metrics["macro_legality"],
                }
            )
            continue

        for step in range(cfg_dict["sa_steps"]):
            temp = cfg_dict["temp_init"] * (cfg_dict["temp_decay"] ** step)
            diff_interval = max(1, int(cfg_dict["diffusion"].get("interval", 1)))
            use_diffusion = cfg_dict["sa_mode"] == "hybrid" and (step % diff_interval == 0)

            if use_diffusion:
                move = maybe_diffusion_move(model, x_cur, cond, cfg_dict["diffusion"])
                if move is None:
                    continue
                x_proposed, _ = move
                move_type = "diffusion"
            else:
                if random.random() < swap_prob:
                    x_proposed = propose_swap(x_cur, macro_idx)
                    move_type = "swap"
                else:
                    x_proposed = propose_jitter(x_cur, macro_idx, jitter_std)
                    move_type = "jitter"
                if x_proposed is None:
                    continue

            new_cost, new_metrics = evaluate_cost(x_proposed, x_ref, cond, cfg_dict["weights"])
            delta = new_cost - cur_cost
            accept = delta <= 0
            if not accept:
                temp_safe = max(temp, 1e-6)
                try:
                    accept = math.exp(-delta / temp_safe) > random.random()
                except OverflowError:
                    accept = False

            if accept:
                x_cur = x_proposed
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_metrics = new_metrics
                    best_x = x_proposed.detach().clone()

            metrics_rows.append(
                {
                    "sample_id": getattr(cond, "file_idx", sample_idx),
                    "step": step,
                    "move": move_type,
                    "temp": temp,
                    "accepted": accept,
                    **new_metrics,
                }
            )

        summaries.append(
            {
                "sample_id": getattr(cond, "file_idx", sample_idx),
                "best_cost": float(best_cost),
                "hpwl_rescaled": best_metrics["hpwl_rescaled"],
                "legality": best_metrics["legality"],
                "macro_legality": best_metrics["macro_legality"],
            }
        )

        sample_id = getattr(cond, "file_idx", sample_idx)
        best_path = os.path.join(samples_dir, f"best{sample_id}.pkl")
        with open(best_path, "wb") as f:
            pickle.dump(best_x.squeeze(0).cpu(), f)
        print(f"Sample {sample_id}: best cost {best_cost:.4f}")

    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": cfg_dict,
                "summaries": summaries,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
