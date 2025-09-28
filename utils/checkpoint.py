# utils/checkpoint.py  (replace load_checkpoint_if_exists with the version below)
import os
from pathlib import Path
import torch
from typing import Optional, Union
from itertools import product

def save_checkpoint(state: dict, file_name: str) -> None:
    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
    torch.save(state, file_name, _use_new_zipfile_serialization=True)


def load_checkpoint(file_name: str, map_location: Optional[Union[dict, str]] = None):
    if not Path(file_name).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_name}")
    if map_location is None:
        map_location = "cpu"
    return torch.load(file_name, map_location=map_location)


def _best_aligned_state_dict(checkpoint_sd: dict, model_keys):
    """
    Try different combinations of stripping/adding 'module.' and '_orig_mod.' prefixes
    and choose the transform that yields the largest key intersection with model_keys.
    Returns transformed_state_dict and a debug dict describing chosen transform.
    """
    ck_keys = list(checkpoint_sd.keys())
    best = {"score": -1, "transform": None, "mapped_keys": None}

    # operations: strip_orig, strip_module, add_module, add_orig (applied in that order)
    for strip_orig, strip_module, add_module, add_orig in product([0, 1], repeat=4):
        mapped = []
        for k in ck_keys:
            k2 = k
            if strip_orig and k2.startswith("_orig_mod."):
                k2 = k2[len("_orig_mod."):]
            if strip_module and k2.startswith("module."):
                k2 = k2[len("module."):]
            if add_module and not k2.startswith("module."):
                k2 = "module." + k2
            if add_orig and not k2.startswith("_orig_mod."):
                k2 = "_orig_mod." + k2
            mapped.append(k2)

        overlap = len(set(mapped) & model_keys)
        if overlap > best["score"]:
            best["score"] = overlap
            best["transform"] = (bool(strip_orig), bool(strip_module), bool(add_module), bool(add_orig))
            best["mapped_keys"] = mapped

    # Apply best transform to build new_state_dict
    if best["score"] <= 0:
        # no overlap found — return original
        return checkpoint_sd, {"chosen": "none", "overlap": 0}

    strip_orig, strip_module, add_module, add_orig = best["transform"]
    new_sd = {}
    for old_k, new_k in zip(ck_keys, best["mapped_keys"]):
        # avoid collisions: prefer existing new key if duplicates appear (last-write wins)
        new_sd[new_k] = checkpoint_sd[old_k]

    debug = {
        "chosen": {
            "strip_orig": strip_orig,
            "strip_module": strip_module,
            "add_module": add_module,
            "add_orig": add_orig
        },
        "overlap": best["score"],
        "ck_size": len(ck_keys),
        "model_key_count": len(model_keys)
    }
    return new_sd, debug


def load_checkpoint_if_exists(model, optimizer, scheduler, scaler, args):
    checkpoint_path = Path(args.resume) if getattr(args, "resume", None) else Path(args.save) / "best_model.pt"

    if not checkpoint_path.exists():
        if getattr(args, "rank0", True):
            print(f"Checkpoint not found at {checkpoint_path}. Starting training from scratch.")
        return 0.0, 0  # start epoch 0

    if getattr(args, "rank0", True):
        print(f"Loading resume checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Allow checkpoint to be either the raw state_dict or a dict with 'model_state_dict'
    ck_model_sd = checkpoint.get('model_state_dict', checkpoint if isinstance(checkpoint, dict) else None)
    if ck_model_sd is None:
        raise RuntimeError("Unable to find model_state_dict in checkpoint.")

    # Try to align keys between checkpoint and model
    model_sd_keys = set(model.state_dict().keys())
    aligned_sd, debug = _best_aligned_state_dict(ck_model_sd, model_sd_keys)

    if getattr(args, "rank0", True):
        print(f"State-dict alignment debug: {debug}")

    # Load model weights
    try:
        model.load_state_dict(aligned_sd, strict=True)
    except RuntimeError as e:
        # Last resort: try non-strict load (will ignore unmatched keys)
        if getattr(args, "rank0", True):
            print(f"Warning: strict load failed ({e}), retrying with strict=False")
        model.load_state_dict(aligned_sd, strict=False)

    # Load optional states with try/except to avoid hard crashes
    def _try_load(name, loader, target):
        if name in checkpoint and target is not None:
            try:
                loader(checkpoint[name])
            except Exception as e:
                if getattr(args, "rank0", True):
                    print(f"Warning: failed to load {name}: {e}")

    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if getattr(args, "rank0", True):
                print(f"Warning: failed to load optimizer state: {e}")

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            if getattr(args, "rank0", True):
                print(f"Warning: failed to load scheduler state: {e}")

    if 'scaler_state_dict' in checkpoint and scaler is not None:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            if getattr(args, "rank0", True):
                print(f"Warning: failed to load scaler state: {e}")

    best_acc = checkpoint.get('best_acc', 0.0)
    start_epoch = checkpoint.get('epoch', 0) + 1

    if getattr(args, "rank0", True):
        print(f"Loaded checkpoint: best_acc={best_acc}, start_epoch={start_epoch}")

    return best_acc, start_epoch
