import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import network
from loss import loss_function
from dataloaders.GSVCities import get_GSVCities
from torch.cuda.amp import GradScaler,autocast

import warnings
warnings.filterwarnings("ignore")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def is_better_checkpoint(candidate_pitts_r1, candidate_msls_r1, best_pitts_r1, best_msls_r1):
    """Compare checkpoints lexicographically: Pitts30k R@1 first, MSLS R@1 second."""
    return (candidate_pitts_r1, candidate_msls_r1) > (best_pitts_r1, best_msls_r1)


#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading datasets {args.eval_dataset_names} from folder {args.eval_datasets_folder}")

eval_datasets = {
    name: datasets_ws.BaseDataset(args, args.eval_datasets_folder, name, "test")
    for name in args.eval_dataset_names
}
for name, ds in eval_datasets.items():
    logging.info(f"Eval set: {name} -> {ds}")

#### Initialize model
model = network.VPRmodel(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

#### Print the number of model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
aggregator_params = sum(p.numel() for p in model.module.aggregator.parameters()) if model.module.aggregator else 0

print(f"The entire parameters: {total_params / 1e6:.2f}M")
print(f"The trainable parameters: {trainable_params / 1e6:.2f}M")
print(f"The aggregator parameters: {aggregator_params / 1e6:.2f}M")

#### Initialize agg tokens
if not args.aggregator:
    args.features_dim = 768
    if not args.resume:
        cache_dir = join(os.getcwd(), "agg_token_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = f"{args.initialization_dataset}_{args.backbone}_tokens{args.num_learnable_aggregation_tokens}_dim{args.features_dim}.npy"
        cache_path = join(cache_dir, cache_key)

        if os.path.exists(cache_path):
            centroids_L2N = np.load(cache_path)
            model.module.learnable_aggregation_tokens = torch.nn.Parameter(
                torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0)
            )
            logging.info(f"Loaded aggregation tokens from cache: {cache_path}")
        else:
            pretrained_model = network.get_backbone(args)
            centroids_L2N = None
            if args.initialization_dataset == "msls_train":
                from initialize_agg_tokens import initialize_learnable_aggregation_tokens_centroids_msls_train, initialize_learnable_aggregation_tokens_centroids_L2N
                triplets_ds = datasets_ws.TripletsDataset(args, args.eval_datasets_folder, "msls", "train", args.negs_num_per_query)
                logging.info(f"Train query set: {triplets_ds}")
                triplets_ds.is_inference = True
                initial_centroids, initial_descriptors = initialize_learnable_aggregation_tokens_centroids_msls_train(args, triplets_ds, pretrained_model.to(args.device))
                centroids_L2N = initialize_learnable_aggregation_tokens_centroids_L2N(initial_centroids, initial_descriptors)
                model.module.learnable_aggregation_tokens = torch.nn.Parameter(torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0))

            elif args.initialization_dataset == "gsv_cities":
                from initialize_agg_tokens import initialize_learnable_aggregation_tokens_centroids_gsv, initialize_learnable_aggregation_tokens_centroids_L2N
                TRAIN_CITIES = [
                'Bangkok',
                'BuenosAires',
                'LosAngeles',
                'MexicoCity',
                'OSL', # refers to Oslo
                'Rome',
                'Barcelona',
                'Chicago',
                'Madrid',
                'Miami',
                'Phoenix',
                'TRT', # refers to Toronto
                'Boston',
                'Lisbon',
                'Medellin',
                'Minneapolis',
                'PRG', # refers to Prague
                'WashingtonDC',
                'Brussels',
                'London',
                'Melbourne',
                'Osaka',
                'PRS', # refers to Paris
                ]
                # initial_dataset = get_GSVCities(image_size=(224, 224), cities=TRAIN_CITIES)
                initial_dataset = get_GSVCities(image_size=(322, 322), cities=TRAIN_CITIES)
                initial_centroids, initial_descriptors = initialize_learnable_aggregation_tokens_centroids_gsv(args, initial_dataset, pretrained_model.to(args.device))
                centroids_L2N = initialize_learnable_aggregation_tokens_centroids_L2N(initial_centroids, initial_descriptors)
                model.module.learnable_aggregation_tokens = torch.nn.Parameter(torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0))

            if centroids_L2N is not None:
                np.save(cache_path, centroids_L2N)
                logging.info(f"Saved aggregation tokens to cache: {cache_path}")
    args.features_dim = args.features_dim * args.num_learnable_aggregation_tokens

if args.aggregator in ["netvlad"]:  # If using NetVLAD layer, initialize it
    args.features_dim = 768
    if not args.resume:
        triplets_ds = datasets_ws.TripletsDataset(args, args.eval_datasets_folder, "msls", "train", args.negs_num_per_query)
        logging.info(f"Train query set: {triplets_ds}")
        triplets_ds.is_inference = True
        pretrained_model = network.get_backbone(args)
        model.module.agg.initialize_netvlad_layer(args, triplets_ds, pretrained_model.to(args.device)) 
    args.features_dim = args.features_dim * 8

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#### Resume model, optimizer, and other training parameters
if args.resume:
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch_num = checkpoint["epoch_num"] + 1
    best_pitts_r1 = float(checkpoint.get("best_pitts_r1", checkpoint.get("pitts_r1", 0.0)))
    best_msls_r1 = float(checkpoint.get("best_msls_r1", checkpoint.get("msls_r1", 0.0)))
    not_improved_num = checkpoint.get("not_improved_num", 0)
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best "
        f"(Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f})"
    )
else:
    best_pitts_r1 = 0.0
    best_msls_r1 = 0.0
    start_epoch_num = not_improved_num = 0

TRAIN_CITIES =[]
args_cities = [city.strip() for city in args.training_dataset.split(",")]
if "gsv_cities" in args_cities:
    args_cities.remove("gsv_cities")
    TRAIN_CITIES = [
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]
for city in args_cities:
    if city not in TRAIN_CITIES:
        TRAIN_CITIES.append(city)

print(f'使用的所有城市包括：{TRAIN_CITIES}')
train_dataset = get_GSVCities(image_size=tuple(args.train_image_size), cities=TRAIN_CITIES)
# train_dataset = get_GSVCities(image_size=(322, 322), cities=TRAIN_CITIES)
train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False}

#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*5, gamma=0.75)
scaler = GradScaler()
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses=[]
    for batch_idx, (images, place_id) in enumerate(tqdm(ds)):
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)

        optimizer.zero_grad()
        with autocast():
            descriptors = model(images.to(args.device)).cuda()
            loss = loss_function(descriptors, labels) # Call the loss_function we defined above
            del descriptors

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        del loss

        if args.fast_debug_batches and batch_idx >= 2:
            logging.info("Fast debug mode enabled: stopping after 3 batches")
            break

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    epoch_recalls = {}
    for eval_dataset_name, eval_ds in eval_datasets.items():
        recalls, recalls_str = test.test(args, eval_ds, model)
        epoch_recalls[eval_dataset_name] = recalls
        if isinstance(recalls_str, dict):
            for subset_name, subset_recalls in recalls_str.items():
                logging.info(f"Recalls on eval set {subset_name}: {subset_recalls}")
        else:
            logging.info(f"Recalls on eval set {eval_dataset_name} {eval_ds}: {recalls_str}")

    for required_name in ("Msls_740", "pitts30k"):
        if required_name not in epoch_recalls:
            raise RuntimeError(
                f"Required eval dataset '{required_name}' is missing from eval_dataset_names"
            )

    msls_r1 = float(epoch_recalls["Msls_740"][0])
    pitts_r1 = float(epoch_recalls["pitts30k"][0])
    is_best = is_better_checkpoint(pitts_r1, msls_r1, best_pitts_r1, best_msls_r1)
    best_state_pitts_r1 = pitts_r1 if is_best else best_pitts_r1
    best_state_msls_r1 = msls_r1 if is_best else best_msls_r1

    # Save checkpoint, which contains all training parameters
    state = {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": epoch_recalls["pitts30k"],
        "eval_recalls": epoch_recalls, "selection_score": [pitts_r1, msls_r1],
        "best_pitts_r1": best_state_pitts_r1,
        "best_msls_r1": best_state_msls_r1,
        "best_msls_pitts_r1": best_state_pitts_r1 + best_state_msls_r1,
        "best_r1_r5": best_state_pitts_r1 + best_state_msls_r1,
        "pitts_r1": pitts_r1, "msls_r1": msls_r1,
        "not_improved_num": not_improved_num
    }
    util.save_checkpoint(args, state, is_best, filename="last_model.pth")
    util.save_epoch_checkpoint(args, state, epoch_recalls["pitts30k"], epoch_num)
    
    # If the checkpoint order (Pitts30k R@1, MSLS R@1) did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            f"Improved: previous best (Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f}), "
            f"current=({pitts_r1:.1f}, {msls_r1:.1f})"
        )
        best_pitts_r1 = pitts_r1
        best_msls_r1 = msls_r1
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: "
            f"best (Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f}), "
            f"current=({pitts_r1:.1f}, {msls_r1:.1f})"
        )
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best checkpoint by (Pitts30k R@1, MSLS R@1): ({best_pitts_r1:.1f}, {best_msls_r1:.1f})")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

# load the best model for testing
logging.info("Test *best* model on test sets")
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"), weights_only=False)["model_state_dict"]
model.load_state_dict(best_model_state_dict)
for eval_dataset_name, eval_ds in eval_datasets.items():
    recalls, recalls_str = test.test(args, eval_ds, model, test_method=args.test_method)
    if isinstance(recalls_str, dict):
        for subset_name, subset_recalls in recalls_str.items():
            logging.info(f"Recalls on {subset_name}: {subset_recalls}")
    else:
        logging.info(f"Recalls on {eval_dataset_name} {eval_ds}: {recalls_str}")
