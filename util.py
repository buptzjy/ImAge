
import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))

def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    model_state = model.state_dict()
    remapped_state = OrderedDict()
    remapped_count = 0
    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith("module."):
            candidates.append(key[len("module."):])
        else:
            candidates.append("module." + key)
        if key.startswith("backbone.dino."):
            tail = key[len("backbone.dino."):]
            candidates.append("backbone." + tail)
            candidates.append("module.backbone." + tail)
        if key.startswith("module.backbone.dino."):
            tail = key[len("module.backbone.dino."):]
            candidates.append("backbone." + tail)
            candidates.append("module.backbone." + tail)
        if key.startswith("backbone.model."):
            tail = key[len("backbone.model."):]
            candidates.append("backbone." + tail)
            candidates.append("module.backbone." + tail)
        if key.startswith("module.backbone.model."):
            tail = key[len("module.backbone.model."):]
            candidates.append("backbone." + tail)
            candidates.append("module.backbone." + tail)

        target_key = None
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                target_key = candidate
                break
        if target_key is None:
            target_key = key
        elif target_key != key:
            remapped_count += 1
        remapped_state[target_key] = value
    if remapped_count:
        logging.info(f"Remapped {remapped_count} checkpoint keys to match the current model")
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    logging.info(
        f"Loaded weights from {args.resume}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, weights_only=False)
    start_epoch_num = checkpoint["epoch_num"]+1
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r1_r5 = checkpoint["best_r1_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current best (R@1 + R@5) = {best_r1_r5:.1f}")
    if args.resume.endswith("best_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("best_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r1_r5, start_epoch_num, not_improved_num

def save_epoch_checkpoint(args, state, recalls, epoch_num):
    recall_map = {v: i for i, v in enumerate(args.recall_values)}
    def _get_r(val):
        idx = recall_map.get(val, None)
        return recalls[idx] if idx is not None else 0.0

    r1 = _get_r(1)
    r5 = _get_r(5)
    r10 = _get_r(10)
    filename = f"Epoch_{epoch_num}_Retrieval_R1_{r1:.2f}_R5_{r5:.2f}_R10_{r10:.2f}_model.pth"
    torch.save(state, join(args.save_dir, filename))
