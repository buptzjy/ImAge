#!/usr/bin/env python3
"""Resume an ImAge baseline and finetune with real-GSV warmup then mixed data.

Schedule:
  - epochs [0, warmup_epochs): real GSV images only
  - epochs [warmup_epochs, ...): real:synthetic images ~= 8:1
  - every epoch: evaluate MSLS, Pitts, SPED, AmsterTime, Tokyo247, Nordland
  - best_model.pth is selected by Pitts R@1 first, then MSLS R@1
"""

import argparse
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import commons  # noqa: E402
import datasets_ws  # noqa: E402
import network  # noqa: E402
import test  # noqa: E402
import util  # noqa: E402
from loss import loss_function  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_better_checkpoint(candidate_pitts_r1, candidate_msls_r1, best_pitts_r1, best_msls_r1):
    """Compare checkpoints lexicographically: Pitts30k R@1 first, MSLS R@1 second."""
    return (candidate_pitts_r1, candidate_msls_r1) > (best_pitts_r1, best_msls_r1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ImAge SALAD/BoQ mixed finetuning from an author/GSV checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aggregator",
        choices=["image", "salad", "boq"],
        required=True,
        help="'image' uses original ImAge learnable aggregation tokens; salad/boq use ImAge aggregators.",
    )
    parser.add_argument("--resume_author", type=str, required=True,
                        help="Author/GSV checkpoint to load as model weights only.")
    parser.add_argument("--training_dataset", type=str,
                        default="/data_nvme/zhangjingyi/Gsv_reflect/mixgsv",
                        help="GSV-style root with Dataframes/ and Images/.")
    parser.add_argument(
        "--training_subsets",
        type=str,
        nargs="+",
        default=["default"],
        help=(
            "Which Dataframes subsets to include. "
            "'default' reads Dataframes/*.csv, and any other value like 'tmp' reads "
            "Dataframes/<subset>/*.csv."
        ),
    )
    parser.add_argument(
        "--tmp_group",
        choices=["all", "pitts", "msls"],
        default="all",
        help=(
            "Optional filter for Dataframes/tmp when 'tmp' is included in training_subsets. "
            "'pitts' keeps tmp/pitts0..pitts17, 'msls' keeps the remaining tmp CSV files."
        ),
    )
    parser.add_argument("--category_file", type=str, default=None,
                        help="Optional TSV: image_path<TAB>{hard,boundary,easy}.")
    parser.add_argument("--policy", choices=["random", "v1", "v2"], default="random",
                        help="Synthetic sampling policy. v1=70/30 hard-boundary/easy, v2=hard-boundary only.")

    parser.add_argument("--eval_datasets_folder", type=str, required=True)
    parser.add_argument("--eval_dataset_names", nargs="+",
                        default=["Msls_740", "pitts30k"])
    parser.add_argument(
        "--final_eval_dataset_names",
        nargs="+",
        default=[],
        help=(
            "Optional datasets to evaluate only once at the end on best_model.pth. "
            "Examples: sped amstertime tokyo nordland svox"
        ),
    )
    parser.add_argument(
        "--occupy_only",
        action="store_true",
        help=(
            "Training-only mode for occupying GPUs. Disables validation selection, "
            "checkpoint saving, and final best-model evaluation."
        ),
    )
    parser.add_argument(
        "--disable_file_logging",
        action="store_true",
        help="Do not write info.log/debug.log files under save_dir.",
    )
    parser.add_argument(
        "--disable_checkpoints",
        action="store_true",
        help="Do not save last_model.pth or best_model.pth checkpoints.",
    )
    parser.add_argument(
        "--disable_final_eval",
        action="store_true",
        help="Skip the final best-model evaluation stage.",
    )
    parser.add_argument("--backbone", type=str, default="dinov2", choices=["dinov2"])
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(0, 11)),
                        help="First trainable transformer block. Defaults: image=8, salad/boq=10.")
    parser.add_argument("--num_register_tokens", type=int, default=None,
                        help="DINOv2 register tokens. Defaults: image=4, salad/boq=0.")
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Only used when resume_author is not a full model checkpoint.")
    parser.add_argument("--num_learnable_aggregation_tokens", type=int, default=8)

    parser.add_argument("--epochs_num", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--real_to_synthetic", type=float, default=8.0)
    parser.add_argument("--train_batch_size", type=int, default=120,
                        help="Number of places per batch.")
    parser.add_argument("--images_per_place", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train_image_size", type=int, nargs=2, default=[322, 322])
    parser.add_argument("--resize", type=int, nargs=2, default=[322, 322])
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--test_method", type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops",
                                 "nearest_crop", "maj_voting"])
    parser.add_argument("--majority_weight", type=float, default=0.01)
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25)
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 100])
    parser.add_argument("--save_dir", type=str, default="mixed_finetune",
                        help="Run name under ImAge/logs/.")
    parser.add_argument("--fast_debug_batches", action="store_true")
    return parser.parse_args()


def set_output_feature_dim(args):
    if not args.aggregator:
        args.features_dim = 768 * args.num_learnable_aggregation_tokens
    elif args.aggregator == "salad":
        args.features_dim = 8448
    elif args.aggregator == "boq":
        args.features_dim = 12288
    else:
        raise ValueError(f"Unsupported aggregator: {args.aggregator}")


def is_synthetic_name(name):
    return "__reflectvpr_" in str(name)


FINAL_EVAL_DATASET_GROUPS = {
    "svox": ["SVOX"],
}

FINAL_EVAL_DATASET_ALIASES = {
    "tokyo": "tokyo247",
    "msls": "Msls_740",
    "msls_740": "Msls_740",
    "svoxnight": "SVOX-night",
    "svoxovercast": "SVOX-overcast",
    "svoxrains": "SVOX-rain",
    "svoxrain": "SVOX-rain",
    "svoxsnow": "SVOX-snow",
    "svoxsun": "SVOX-sun",
}


def expand_eval_dataset_names(dataset_names):
    expanded = []
    for name in dataset_names:
        normalized = name.lower().replace("-", "_")
        compact = normalized.replace("_", "")
        canonical = FINAL_EVAL_DATASET_ALIASES.get(compact, FINAL_EVAL_DATASET_ALIASES.get(normalized, name))
        if canonical in FINAL_EVAL_DATASET_GROUPS:
            expanded.extend(FINAL_EVAL_DATASET_GROUPS[canonical])
        else:
            expanded.append(canonical)

    ordered = []
    seen = set()
    for name in expanded:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def infer_eval_split(dataset_name):
    return "val" if dataset_name.lower() == "msls_740" else "test"


def load_categories(path):
    if path is None:
        return None
    categories = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_path, category = line.split("\t")
            if category in {"hard", "boundary", "easy"}:
                categories[os.path.realpath(image_path)] = category
    if not categories:
        raise RuntimeError(f"No usable synthetic categories in {path}")
    return categories


class WarmupMixedGSVDataset(Dataset):
    def __init__(self, base_path, category_file=None, images_per_place=4,
                 image_size=(322, 322), seed=0, training_subsets=None, tmp_group="all"):
        self.base_path = Path(base_path).expanduser().resolve()
        self.images_per_place = images_per_place
        self.seed = seed
        self.tmp_group = tmp_group
        self.categories = load_categories(category_file)
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        dataframes_dir = self.base_path / "Dataframes"
        self.training_subsets = training_subsets or ["default"]
        self.csv_sources = self.collect_csv_sources(
            dataframes_dir,
            self.training_subsets,
            tmp_group=self.tmp_group,
        )
        if not self.csv_sources:
            raise FileNotFoundError(
                f"No CSV files found in {dataframes_dir} for subsets {self.training_subsets}"
            )
        self.cities = [source["city_name"] for source in self.csv_sources]

        self.places = []
        self.pool_by_group = defaultdict(list)
        category_counts = Counter()
        total_real = 0
        total_synthetic = 0
        skipped_missing_real = 0
        skipped_missing_synthetic = 0

        for city_index, source in enumerate(self.csv_sources):
            city = source["city_name"]
            dataframe = pd.read_csv(source["csv_path"])
            for local_place, rows in dataframe.groupby("place_id", sort=False):
                real_paths = []
                synthetic_by_category = defaultdict(list)
                synthetic_all = []
                for row in rows.itertuples(index=False):
                    image_name = self.image_name(row)
                    image_path = source["images_dir"] / str(row.city_id) / image_name
                    if is_synthetic_name(image_name):
                        if not image_path.is_file():
                            skipped_missing_synthetic += 1
                            continue
                        synthetic_all.append(str(image_path))
                        total_synthetic += 1
                        if self.categories is not None:
                            category = self.categories.get(os.path.realpath(image_path))
                            if category is not None:
                                synthetic_by_category[category].append(str(image_path))
                                category_counts[category] += 1
                    else:
                        if not image_path.is_file():
                            skipped_missing_real += 1
                            continue
                        real_paths.append(str(image_path))
                        total_real += 1

                if len(real_paths) < images_per_place:
                    continue
                if self.categories is None:
                    synthetic_by_category["any"] = synthetic_all

                place_index = len(self.places)
                global_label = city_index * 10_000_000 + int(local_place)
                self.places.append({
                    "label": global_label,
                    "real": real_paths,
                    "synthetic": dict(synthetic_by_category),
                })
                self.pool_by_group["real"].append(place_index)
                if synthetic_all:
                    self.pool_by_group["any"].append(place_index)
                if synthetic_by_category.get("hard") or synthetic_by_category.get("boundary"):
                    self.pool_by_group["hard_boundary"].append(place_index)
                if synthetic_by_category.get("easy"):
                    self.pool_by_group["easy"].append(place_index)

        self.total_nb_images = total_real + total_synthetic
        self.category_counts = category_counts
        self.skipped_missing_real = skipped_missing_real
        self.skipped_missing_synthetic = skipped_missing_synthetic
        if not self.pool_by_group["real"]:
            raise RuntimeError("No real places with enough images were found")
        if not self.pool_by_group["any"] and not self.pool_by_group["hard_boundary"]:
            raise RuntimeError("No synthetic places were found for mixed finetuning")

    @staticmethod
    def normalize_training_subsets(training_subsets):
        normalized = []
        for subset in training_subsets or ["default"]:
            subset_name = subset.strip()
            if not subset_name:
                continue
            normalized.append("default" if subset_name in {".", "root"} else subset_name)
        return normalized or ["default"]

    @classmethod
    def is_tmp_pitts_csv(cls, csv_path):
        stem = csv_path.stem.strip().lower()
        if not stem.startswith("pitts"):
            return False
        suffix = stem[len("pitts"):]
        return suffix.isdigit() and 0 <= int(suffix) <= 17

    @classmethod
    def should_include_csv(cls, subset, csv_path, tmp_group):
        if subset != "tmp" or tmp_group == "all":
            return True
        is_pitts_csv = cls.is_tmp_pitts_csv(csv_path)
        if tmp_group == "pitts":
            return is_pitts_csv
        if tmp_group == "msls":
            return not is_pitts_csv
        return True

    @classmethod
    def collect_csv_sources(cls, dataframes_dir, training_subsets, tmp_group="all"):
        normalized_subsets = cls.normalize_training_subsets(training_subsets)
        excluded_cities = {"tmp_pitts"}
        csv_sources = []
        seen_keys = set()

        for subset in normalized_subsets:
            subset_dir = dataframes_dir if subset == "default" else dataframes_dir / subset
            if not subset_dir.is_dir():
                logging.warning(f"Training subset directory not found, skipping: {subset_dir}")
                continue

            images_root = dataframes_dir.parent / "Images"
            subset_images_dir = images_root if subset == "default" else images_root / subset
            active_images_dir = subset_images_dir if subset_images_dir.is_dir() else images_root

            for csv_path in sorted(subset_dir.glob("*.csv")):
                if csv_path.stem in excluded_cities:
                    continue
                if not cls.should_include_csv(subset, csv_path, tmp_group):
                    continue
                source_key = (subset, csv_path.stem)
                if source_key in seen_keys:
                    continue
                seen_keys.add(source_key)
                city_name = csv_path.stem if subset == "default" else f"{subset}/{csv_path.stem}"
                csv_sources.append({
                    "subset": subset,
                    "city_name": city_name,
                    "csv_path": csv_path,
                    "images_dir": active_images_dir,
                })

        return csv_sources

    @staticmethod
    def image_name(row):
        place_id = str(int(row.place_id)).zfill(7)
        year = str(int(row.year)).zfill(4)
        month = str(int(row.month)).zfill(2)
        northdeg = str(int(row.northdeg)).zfill(3)
        return (
            f"{row.city_id}_{place_id}_{year}_{month}_{northdeg}_"
            f"{row.lat}_{row.lon}_{row.panoid}.jpg"
        )

    @staticmethod
    def load_image(path):
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            logging.warning(f"Image could not be loaded: {path}")
            return Image.new("RGB", (322, 322))

    def __len__(self):
        return len(self.places)

    def __getitem__(self, request):
        if isinstance(request, (tuple, list)):
            place_index, group, draw_seed = request
        else:
            place_index, group, draw_seed = int(request), "real", self.seed + int(request)

        rng = random.Random(int(draw_seed))
        place = self.places[int(place_index)]
        real_paths = place["real"]
        images = []

        if group == "real":
            selected = rng.sample(real_paths, self.images_per_place)
        else:
            synth_pool = []
            if group == "hard_boundary":
                synth_pool.extend(place["synthetic"].get("hard", []))
                synth_pool.extend(place["synthetic"].get("boundary", []))
            elif group == "easy":
                synth_pool.extend(place["synthetic"].get("easy", []))
            else:
                for values in place["synthetic"].values():
                    synth_pool.extend(values)
            if not synth_pool:
                selected = rng.sample(real_paths, self.images_per_place)
            else:
                selected = rng.sample(real_paths, self.images_per_place - 1)
                selected.append(rng.choice(synth_pool))
                rng.shuffle(selected)

        for path in selected:
            images.append(self.transform(self.load_image(path)))
        labels = torch.tensor(place["label"]).repeat(self.images_per_place)
        return torch.stack(images), labels


class WarmupMixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, warmup_epochs=3,
                 real_to_synthetic=8.0, policy="random", seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.real_to_synthetic = real_to_synthetic
        self.policy = policy
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset.pool_by_group["real"]) // self.batch_size

    def _choose_group(self, rng):
        if self.policy == "v1":
            return "hard_boundary" if rng.random() < 0.7 else "easy"
        if self.policy == "v2":
            return "hard_boundary"
        return "any"

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        real_pool = self.dataset.pool_by_group["real"][:]
        rng.shuffle(real_pool)
        real_cursor = 0

        synth_places_per_batch = 0
        if self.epoch >= self.warmup_epochs:
            synth_images = self.batch_size * self.dataset.images_per_place / (self.real_to_synthetic + 1.0)
            synth_places_per_batch = int(round(synth_images))
            synth_places_per_batch = max(1, min(self.batch_size, synth_places_per_batch))

        for batch_index in range(len(self)):
            requests = []
            used_places = set()

            for synth_index in range(synth_places_per_batch):
                group = self._choose_group(rng)
                pool = self.dataset.pool_by_group.get(group) or self.dataset.pool_by_group["any"]
                place_index = rng.choice(pool)
                used_places.add(place_index)
                requests.append((
                    place_index,
                    group,
                    self.seed + self.epoch * 10**9 + batch_index * 10**5 + synth_index,
                ))

            while len(requests) < self.batch_size:
                if real_cursor >= len(real_pool):
                    rng.shuffle(real_pool)
                    real_cursor = 0
                place_index = real_pool[real_cursor]
                real_cursor += 1
                if place_index in used_places:
                    continue
                used_places.add(place_index)
                requests.append((
                    place_index,
                    "real",
                    self.seed + self.epoch * 10**9 + batch_index * 10**5 + len(requests),
                ))
            rng.shuffle(requests)
            yield requests


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_state = model.state_dict()
    remapped_state = {}
    remapped_count = 0
    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith("backbone.dino."):
            tail = key[len("backbone.dino."):]
            candidates.append("backbone." + tail)
            candidates.append("module.backbone." + tail)
        if key.startswith("module.backbone.dino."):
            tail = key[len("module.backbone.dino."):]
            candidates.append("backbone." + tail)
            candidates.append("module.backbone." + tail)
        if key.startswith("backbone.model."):
            candidates.append("module.backbone." + key[len("backbone.model."):])
        if key.startswith("module.backbone.model."):
            candidates.append("module.backbone." + key[len("module.backbone.model."):])
        if key.startswith("aggregator."):
            candidates.append("module." + key)
        if not key.startswith("module."):
            candidates.append("module." + key)

        target_key = key
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                target_key = candidate
                break
        if target_key != key:
            remapped_count += 1
        remapped_state[target_key] = value

    if remapped_count:
        logging.info(f"Remapped {remapped_count} checkpoint keys to match the current model")
        state_dict = remapped_state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(
        f"Loaded weights from {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def load_checkpoint_state_dict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def log_eval_results(eval_name, eval_ds, recalls, recalls_str, prefix="Recalls on"):
    if isinstance(recalls_str, dict):
        for subset_name, subset_recalls in recalls_str.items():
            logging.info(f"{prefix} {eval_name}/{subset_name} {eval_ds}: {subset_recalls}")
    else:
        logging.info(f"{prefix} {eval_name} {eval_ds}: {recalls_str}")


def run_final_best_model_eval(args, model):
    final_eval_dataset_names = expand_eval_dataset_names(args.final_eval_dataset_names)
    if not final_eval_dataset_names:
        return

    best_model_path = join(args.save_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        logging.warning(f"Skipping final best-model evaluation because {best_model_path} does not exist")
        return

    logging.info(
        f"Starting final best-model evaluation on {len(final_eval_dataset_names)} datasets: "
        f"{', '.join(final_eval_dataset_names)}"
    )
    load_checkpoint_state_dict(model, best_model_path, args.device)
    model.eval()

    final_eval_datasets = {
        name: datasets_ws.BaseDataset(args, args.eval_datasets_folder, name, infer_eval_split(name))
        for name in final_eval_dataset_names
    }

    for eval_name, eval_ds in final_eval_datasets.items():
        logging.info(f"Final eval set: {eval_name} -> {eval_ds}")
        recalls, recalls_str = test.test(args, eval_ds, model, test_method=args.test_method)
        log_eval_results(eval_name, eval_ds, recalls, recalls_str, prefix="Final recalls on")


def main():
    args = parse_args()
    requested_aggregator = args.aggregator
    if requested_aggregator == "image":
        args.aggregator = None
    if args.freeze_te is None:
        args.freeze_te = 8 if requested_aggregator == "image" else 10
    if args.num_register_tokens is None:
        args.num_register_tokens = 4 if requested_aggregator == "image" else 0
    set_output_feature_dim(args)
    start_time = datetime.now()
    args.resume = args.resume_author
    args.save_dir = str(ROOT / "logs" / args.save_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    if args.occupy_only:
        args.disable_checkpoints = True
        args.disable_final_eval = True
        args.eval_dataset_names = []
        args.final_eval_dataset_names = []
    commons.setup_logging(
        args.save_dir,
        info_filename=None if args.disable_file_logging else "info.log",
        debug_filename=None if args.disable_file_logging else "debug.log",
    )
    commons.make_deterministic(args.seed)
    args.eval_dataset_names = expand_eval_dataset_names(args.eval_dataset_names)
    args.final_eval_dataset_names = expand_eval_dataset_names(args.final_eval_dataset_names)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")

    eval_datasets = {
        name: datasets_ws.BaseDataset(args, args.eval_datasets_folder, name, "test")
        for name in args.eval_dataset_names
    }
    for name, ds in eval_datasets.items():
        logging.info(f"Eval set: {name} -> {ds}")

    model = network.VPRmodel(args).to(args.device)
    logging.info(f"Output dimension of the model is {args.features_dim}")
    model = torch.nn.DataParallel(model)
    load_model_weights(model, args.resume_author, args.device)

    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    scaler = GradScaler()

    train_dataset = WarmupMixedGSVDataset(
        args.training_dataset,
        category_file=args.category_file,
        images_per_place=args.images_per_place,
        image_size=tuple(args.train_image_size),
        seed=args.seed,
        training_subsets=args.training_subsets,
        tmp_group=args.tmp_group,
    )
    sampler = WarmupMixedBatchSampler(
        train_dataset,
        batch_size=args.train_batch_size,
        warmup_epochs=args.warmup_epochs,
        real_to_synthetic=args.real_to_synthetic,
        policy=args.policy,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logging.info(
        f"Training set: root={args.training_dataset}, subsets={args.training_subsets}, "
        f"tmp_group={args.tmp_group}, "
        f"cities={len(train_dataset.cities)}, "
        f"places={len(train_dataset)}, images={train_dataset.total_nb_images}, "
        f"synthetic_categories={dict(train_dataset.category_counts)}, "
        f"skipped_missing_real={train_dataset.skipped_missing_real}, "
        f"skipped_missing_synthetic={train_dataset.skipped_missing_synthetic}"
    )

    best_pitts_r1 = 0.0
    best_msls_r1 = 0.0
    not_improved_num = 0
    for epoch_num in range(args.epochs_num):
        sampler.set_epoch(epoch_num)
        phase = "real_only" if epoch_num < args.warmup_epochs else f"mixed_real_to_synthetic_{args.real_to_synthetic:g}:1"
        logging.info(f"Start training epoch {epoch_num:02d}: phase={phase}")
        model.train()
        epoch_losses = []

        for batch_idx, (images, place_id) in enumerate(tqdm(train_loader)):
            batch_size, num_images, channels, height, width = images.shape
            images = images.view(batch_size * num_images, channels, height, width)
            labels = place_id.view(-1)

            optimizer.zero_grad()
            with autocast():
                descriptors = model(images.to(args.device, non_blocking=True)).cuda()
                loss, batch_acc = loss_function(descriptors, labels.to(args.device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))

            if args.fast_debug_batches and batch_idx >= 2:
                logging.info("Fast debug mode enabled: stopping after 3 batches")
                break

        logging.info(f"Finished epoch {epoch_num:02d}, average loss={np.mean(epoch_losses):.4f}")

        if args.occupy_only:
            continue

        epoch_recalls = {}
        for eval_name, eval_ds in eval_datasets.items():
            recalls, recalls_str = test.test(args, eval_ds, model, test_method=args.test_method)
            epoch_recalls[eval_name] = recalls
            log_eval_results(eval_name, eval_ds, recalls, recalls_str)

        msls_r1 = float(epoch_recalls["Msls_740"][0])
        pitts_r1 = float(epoch_recalls["pitts30k"][0])
        is_best = is_better_checkpoint(pitts_r1, msls_r1, best_pitts_r1, best_msls_r1)
        best_state_pitts_r1 = pitts_r1 if is_best else best_pitts_r1
        best_state_msls_r1 = msls_r1 if is_best else best_msls_r1
        state = {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eval_recalls": epoch_recalls,
            "selection_score": [pitts_r1, msls_r1],
            "best_pitts_r1": best_state_pitts_r1,
            "best_msls_r1": best_state_msls_r1,
            "best_msls_pitts_r1": best_state_pitts_r1 + best_state_msls_r1,
            "msls_r1": msls_r1,
            "pitts_r1": pitts_r1,
            "not_improved_num": not_improved_num,
        }
        if not args.disable_checkpoints:
            util.save_checkpoint(args, state, is_best, filename="last_model.pth")

        if is_best:
            logging.info(
                f"Improved: best (Pitts R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f}) "
                f"-> ({pitts_r1:.1f}, {msls_r1:.1f})"
            )
            best_pitts_r1 = pitts_r1
            best_msls_r1 = msls_r1
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num}/{args.patience}; "
                f"best=({best_pitts_r1:.1f}, {best_msls_r1:.1f}), "
                f"current=({pitts_r1:.1f}, {msls_r1:.1f})"
            )
            if not_improved_num >= args.patience:
                logging.info("Early stopping.")
                break

    if not args.occupy_only:
        logging.info(f"Best checkpoint by (Pitts R@1, MSLS R@1): ({best_pitts_r1:.1f}, {best_msls_r1:.1f})")
    if not args.disable_final_eval:
        run_final_best_model_eval(args, model)
    logging.info(f"Total time: {str(datetime.now() - start_time)[:-7]}")


if __name__ == "__main__":
    main()
