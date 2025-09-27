# utils/checkpoint.py
import os
from pathlib import Path

import torch
from typing import Optional

def save_checkpoint(state: dict, file_name: str) -> None:
    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
    torch.save(state, file_name, _use_new_zipfile_serialization=True)


def load_checkpoint(file_name: str, map_location: Optional[dict or str] = None):
    if map_location is None:
        map_location = "cpu"
    return torch.load(file_name, map_location=map_location)


def load_checkpoint_if_exists(model, optimizer, scheduler, scaler, args):
    """Load checkpoint if exists, handling compiled models."""
    checkpoint_path = Path(args.resume) if args.resume else Path(args.save) / "best_model.pt"

    if not checkpoint_path.exists():
        return 0.0, 1  # Start from scratch

    if args.rank0:
        print(f"Loading resume checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle state dict for compiled models
    state_dict = checkpoint['model_state_dict']

    # Remove _orig_mod prefix if present (for compiled models)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Load state dict with strict=False to handle architectural changes
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        if args.rank0:
            print(f"Warning: strict load failed, retrying with strict=False: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    # Load optimizer, scheduler, scaler states if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    best_acc = checkpoint.get('best_acc', 0.0)
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch

    if args.rank0:
        print(f"Loaded checkpoint: best_acc={best_acc}, start_epoch={start_epoch}")

    return best_acc, start_epoch
