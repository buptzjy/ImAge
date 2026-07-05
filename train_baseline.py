import torch
import logging
import numpy as np
import atexit
import signal
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
import dataloaders.train.GSVCitiesDataset as gsv_dataset_module
from torch.cuda.amp import GradScaler,autocast

import warnings
warnings.filterwarnings("ignore")
import os
### 🌟 新增：引入 TensorBoard
from torch.utils.tensorboard import SummaryWriter
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir, info_filename="log.txt")
commons.make_deterministic(args.seed)

### 🌟 新增：初始化 TensorBoard 的 Writer，保存在当前的 log 文件夹
writer = SummaryWriter(log_dir=args.save_dir)

run_id = start_time.strftime("%Y%m%d_%H%M%S")
log_footer_path = join(args.save_dir, "log.txt")
epoch_records = []
best_epoch_record = None
footer_written = False
best_pitts_r1_r5 = 0.0
best_msls_r1_r5 = 0.0
best_pitts_r1 = 0.0
best_msls_r1 = 0.0

def recall_at(recalls, recall_value, default=0.0):
    recall_map = {v: i for i, v in enumerate(args.recall_values)}
    idx = recall_map.get(recall_value)
    if idx is None or idx >= len(recalls):
        return default
    return float(recalls[idx])

def r1_plus_r5(recalls):
    return recall_at(recalls, 1) + recall_at(recalls, 5)

def is_better_checkpoint(candidate_pitts_r1, candidate_msls_r1, current_best_pitts_r1, current_best_msls_r1):
    """Compare checkpoints lexicographically: Pitts30k R@1 first, MSLS R@1 second."""
    return (candidate_pitts_r1, candidate_msls_r1) > (current_best_pitts_r1, current_best_msls_r1)

def try_save_checkpoint(save_fn, *save_args):
    try:
        save_fn(*save_args)
        return True
    except (OSError, RuntimeError) as e:
        logging.error(f"Checkpoint save failed: {e}")
        logging.error("Training will continue, but free disk space or change --save_dir before relying on checkpoints.")
        return False

def append_log_footer(exit_reason="finished"):
    global footer_written
    if footer_written:
        return
    footer_written = True

    param_keys = [
        "train_batch_size", "lr", "optim", "epochs_num", "patience",
        "backbone", "freeze_te", "num_learnable_aggregation_tokens",
        "training_dataset", "initialization_dataset", "foundation_model_path",
        "resize", "train_image_size", "test_method", "eval_dataset_name",
        "eval_dataset_names", "fast_debug_batches", "save_epoch_checkpoints", "seed"
    ]
    lines = [
        "",
        "=" * 88,
        "EXPERIMENT SUMMARY",
        f"run_id: {run_id}",
        f"exit_reason: {exit_reason}",
        f"save_dir: {args.save_dir}",
        "",
        "[PARAMS]",
    ]
    for key in param_keys:
        lines.append(f"{key}: {getattr(args, key, None)}")

    lines.extend([
        "",
        "[EPOCH METRICS]",
        f"epoch | loss | pitts R@1/R@5/R@10/R@100 | MSLS R@1/R@5/R@10/R@100 | {args.eval_dataset_name} R@1+R@5 | best",
    ])
    if epoch_records:
        for record in epoch_records:
            lines.append(
                f"{record['epoch']:>5} | "
                f"{record['avg_triplet_loss']:.4f} | "
                f"{record['pitts_r1']:.1f}/{record['pitts_r5']:.1f}/{record['pitts_r10']:.1f}/{record['pitts_r100']:.1f} | "
                f"{record['msls_r1']:.1f}/{record['msls_r5']:.1f}/{record['msls_r10']:.1f}/{record['msls_r100']:.1f} | "
                f"{record['primary_r1_plus_r5']:.1f} | "
                f"{'yes' if record['is_best'] else 'no'}"
            )
    else:
        lines.append("No completed validation epoch yet.")

    if best_epoch_record is not None:
        lines.extend([
            "",
            "[BEST]",
            f"epoch: {best_epoch_record['epoch']}",
            f"{args.eval_dataset_name} R@1/R@5: {best_epoch_record['primary_r1']:.1f}/{best_epoch_record['primary_r5']:.1f}",
            f"{args.eval_dataset_name} R@1+R@5: {best_epoch_record['primary_r1_plus_r5']:.1f}",
            f"pitts R@1/R@5: {best_epoch_record['pitts_r1']:.1f}/{best_epoch_record['pitts_r5']:.1f}",
        ])
    lines.append("=" * 88)

    with open(log_footer_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def handle_exit_signal(signum, frame):
    append_log_footer(f"interrupted by signal {signum}")
    raise SystemExit(128 + signum)

atexit.register(append_log_footer)
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

BASE_TRAIN_CITIES = [
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

def get_unified_train_cities():
    train_cities = list(BASE_TRAIN_CITIES)
    directory = '/data_nvme/zhangjingyi/Unified_cities/Dataframes'
    csv_files_no_ext = [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith('.csv')
    ]
    train_cities.extend(csv_files_no_ext)
    return list(dict.fromkeys(train_cities))

def is_dataset_root(value):
    if not value:
        return False
    return os.path.isdir(value) and os.path.isdir(os.path.join(value, "Dataframes")) and os.path.isdir(os.path.join(value, "Images"))

def activate_dataset_root(dataset_root):
    dataset_root = os.path.abspath(dataset_root)
    gsv_dataset_module.ROOT_PATH_LIST[:] = [
        path for path in gsv_dataset_module.ROOT_PATH_LIST
        if os.path.abspath(path) != dataset_root
    ]
    gsv_dataset_module.ROOT_PATH_LIST.insert(0, dataset_root)
    logging.info(f"Using dataset root with highest priority: {dataset_root}")

def get_train_cities_from_dataset_root(dataset_root):
    dataframes_dir = os.path.join(dataset_root, "Dataframes")
    csv_files_no_ext = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(dataframes_dir)
        if f.endswith(".csv")
    )
    if not csv_files_no_ext:
        raise FileNotFoundError(f"No CSV files found in {dataframes_dir}")
    return csv_files_no_ext

def safe_cache_component(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "dataset"

#### Creation of Datasets
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

primary_eval_name = args.eval_dataset_name if args.eval_dataset_name in args.eval_dataset_names else args.eval_dataset_names[0]
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
        cache_key = f"{safe_cache_component(args.initialization_dataset)}_{args.backbone}_tokens{args.num_learnable_aggregation_tokens}_dim{args.features_dim}.npy"
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
                TRAIN_CITIES = list(BASE_TRAIN_CITIES)
                # initial_dataset = get_GSVCities(image_size=(224, 224), cities=TRAIN_CITIES)
                initial_dataset = get_GSVCities(image_size=(322, 322), cities=TRAIN_CITIES)
                initial_centroids, initial_descriptors = initialize_learnable_aggregation_tokens_centroids_gsv(args, initial_dataset, pretrained_model.to(args.device))
                centroids_L2N = initialize_learnable_aggregation_tokens_centroids_L2N(initial_centroids, initial_descriptors)
                model.module.learnable_aggregation_tokens = torch.nn.Parameter(torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0))

            elif args.initialization_dataset in ["with_gen_Unified_cities", "unified_dataset", "unified_cities"]:
                from initialize_agg_tokens import initialize_learnable_aggregation_tokens_centroids_gsv, initialize_learnable_aggregation_tokens_centroids_L2N
                TRAIN_CITIES = get_unified_train_cities()
                initial_dataset = get_GSVCities(image_size=tuple(args.train_image_size), cities=TRAIN_CITIES)
                logging.info(f"Initializing aggregation tokens from unified dataset with {len(TRAIN_CITIES)} cities")
                initial_centroids, initial_descriptors = initialize_learnable_aggregation_tokens_centroids_gsv(args, initial_dataset, pretrained_model.to(args.device))
                centroids_L2N = initialize_learnable_aggregation_tokens_centroids_L2N(initial_centroids, initial_descriptors)
                model.module.learnable_aggregation_tokens = torch.nn.Parameter(torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0))

            elif is_dataset_root(args.initialization_dataset):
                from initialize_agg_tokens import initialize_learnable_aggregation_tokens_centroids_gsv, initialize_learnable_aggregation_tokens_centroids_L2N
                activate_dataset_root(args.initialization_dataset)
                TRAIN_CITIES = get_train_cities_from_dataset_root(args.initialization_dataset)
                initial_dataset = get_GSVCities(image_size=tuple(args.train_image_size), cities=TRAIN_CITIES)
                logging.info(
                    f"Initializing aggregation tokens from dataset root {args.initialization_dataset} "
                    f"with {len(TRAIN_CITIES)} cities"
                )
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
    model, optimizer, best_r1_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r1_r5:.1f}")
    resume_checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    best_pitts_r1_r5 = float(resume_checkpoint.get("best_pitts_r1_r5", 0.0))
    best_msls_r1_r5 = float(resume_checkpoint.get("best_msls_r1_r5", 0.0))
    best_pitts_r1 = float(resume_checkpoint.get("best_pitts_r1", resume_checkpoint.get("pitts_r1", 0.0)))
    best_msls_r1 = float(resume_checkpoint.get("best_msls_r1", resume_checkpoint.get("msls_r1", 0.0)))
    logging.info(
        f"Checkpoint selection state resumed as "
        f"(Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f})"
    )
else:
    best_r1_r5 = start_epoch_num = not_improved_num = 0

print("args.training_dataset: ", args.training_dataset)

if args.training_dataset == "gsv_cities":
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
        'Phoenix_with_gen',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London_with_gen',
        'Melbourne',
        'Osaka_with_gen',
        'PRS', # refers to Paris
    ]
elif args.training_dataset in ["with_gen_Unified_cities", "unified_dataset", "unified_cities"]:
    TRAIN_CITIES = get_unified_train_cities()
elif is_dataset_root(args.training_dataset):
    activate_dataset_root(args.training_dataset)
    TRAIN_CITIES = get_train_cities_from_dataset_root(args.training_dataset)
else:
    TRAIN_CITIES = ['London_with_gen','Osaka_with_gen','Phoenix_with_gen']
    # TRAIN_CITIES = [
    #     "SFXL",
    #     'Bangkok',
    #     'BuenosAires',
    #     'LosAngeles',
    #     'MexicoCity',
    #     'OSL', # refers to Oslo
    #     'Rome',
    #     'Barcelona',
    #     'Chicago',
    #     'Madrid',
    #     'Miami',
    #     'Phoenix',
    #     'TRT', # refers to Toronto
    #     'Boston',
    #     'Lisbon',
    #     'Medellin',
    #     'Minneapolis',
    #     'PRG', # refers to Prague
    #     'WashingtonDC',
    #     'Brussels',
    #     'London',
    #     'Melbourne',
    #     'Osaka',
    #     'PRS', # refers to Paris
    # ]
    # citylist = [
    #     "Trondheim",
    #     "Amsterdam",
    #     "Helsinki",
    #     "Tokyo",
    #     "Toronto",
    #     "Saopaulo",
    #     "Moscow",
    #     "Zurich",
    #     "Paris",
    #     "Budapest",
    #     "Austin",
    #     "Berlin",
    #     "Ottawa",
    #     "Goa",
    #     "Amman",
    #     "Nairobi",
    #     "Manila",
    #     "bangkok",
    #     "boston",
    #     "london",
    #     "melbourne",
    #     "phoenix",
    #     "Pitts30k"
    # ]

    # TRAIN_CITIES = [
    #     "SFXL",
    #     'Bangkok',
    #     'BuenosAires',
    #     'LosAngeles',
    #     'MexicoCity',
    #     'OSL',  # refers to Oslo
    #     'Rome',
    #     'Barcelona',
    #     'Chicago',
    #     'Madrid',
    #     'Miami',
    #     'Phoenix',
    #     'TRT',  # refers to Toronto
    #     'Boston',
    #     'Lisbon',
    #     'Medellin',
    #     'Minneapolis',
    #     'PRG',  # refers to Prague
    #     'WashingtonDC',
    #     'Brussels',
    #     'London',
    #     'Melbourne',
    #     'Osaka',
    #     'PRS',  # refers to Paris
    # ]
    # citylist = [
    #     "trondheim",
    #     "amsterdam",
    #     "helsinki",
    #     "tokyo",
    #     "toronto",
    #     "saopaulo",
    #     "moscow",
    #     "zurich",
    #     "paris",
    #     "budapest",
    #     "austin",
    #     "berlin",
    #     "ottawa",
    #     "goa",
    #     "amman",
    #     "nairobi",
    #     "manila",
    #     "bangkok",
    #     "boston",
    #     "london",
    #     "melbourne",
    #     "phoenix",
    #     "Pitts30k"
    # ]
    # newcitylist = []
    # for i in range(18):
    #     for cityname in citylist:
    #         # if i==17 and (cityname == "Amman" or cityname == "Nairobi"):
    #         if i==17 and (cityname == "amman" or cityname == "nairobi"):
    #             continue
    #         else:
    #             newcitylist.append(cityname+str(i))
    # TRAIN_CITIES = TRAIN_CITIES + newcitylist

# train_dataset = get_GSVCities(image_size=(224, 224), cities=TRAIN_CITIES)
print('TRAIN_CITIES: ', TRAIN_CITIES)
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
    # epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses=[]
   
    # 🌟 修改 1：定义 tqdm 对象，desc 可以写 Epoch 号
    pbar = tqdm(enumerate(ds), total=len(ds), desc=f"Epoch {epoch_num:02d}")

    for batch_idx, (images, place_id, _) in pbar:       
        BS, N, ch, h, w = images.shape
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)

        optimizer.zero_grad()
        with autocast():
            descriptors = model(images.to(args.device))
            
            # 🌟 修改 2：这里接收两个返回值
            loss, b_acc = loss_function(descriptors, labels) 
            
            del descriptors

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        batch_loss = loss.item()
        epoch_losses.append(batch_loss)

        # 🌟 修改 3：实时更新进度条右侧数值
        pbar.set_postfix({
            'b_acc': f"{b_acc:.4f}",
            'loss': f"{batch_loss:.4f}"
        })

        # 🌟 修改 4：实现截图里那种“每 1% 刷一行”的效果
        # 如果当前 batch 刚好是总数的 1% 的倍数，就 print 一行
        if (batch_idx + 1) % max(1, (len(ds) // 100)) == 0:
            # 获取当前的进度条字符串
            progress_msg = f"Epoch {epoch_num}: {int((batch_idx+1)/len(ds)*100):>3d}% || {pbar}"
            # 这里打印会导致屏幕“刷屏”，效果就和你截图一模一样了
            print(progress_msg)

        del loss

        if args.fast_debug_batches and batch_idx >= 2:
            logging.info("Fast debug mode enabled: stopping after 3 batches")
            break

    # ... 后面 logging.info 等代码保持不变 ...

    # for images, place_id, _ in tqdm(ds):       
    #     BS, N, ch, h, w = images.shape
    #     # reshape places and labels
    #     images = images.view(BS*N, ch, h, w)
    #     labels = place_id.view(-1)

    #     optimizer.zero_grad()
    #     with autocast():
    #         descriptors = model(images.to(args.device)).cuda()
    #         loss = loss_function(descriptors, labels) # Call the loss_function we defined above
    #         del descriptors

    #     scaler.scale(loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    #     scheduler.step()
        
    #     # Keep track of all losses by appending them to epoch_losses
    #     batch_loss = loss.item()
    #     epoch_losses = np.append(epoch_losses, batch_loss)
    #     del loss

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {np.mean(epoch_losses):.4f}")
    
    ### 🌟 新增：写入学习率和平均 Loss 到 TensorBoard
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Train/LearningRate', current_lr, epoch_num)
    writer.add_scalar('Train/AverageLoss', np.mean(epoch_losses), epoch_num)

    
    eval_recalls = {}
    primary_recalls = None
    for eval_dataset_name, eval_ds in eval_datasets.items():
        recalls, recalls_str = test.test(args, eval_ds, model)
        eval_recalls[eval_dataset_name] = recalls
        if isinstance(recalls_str, dict):
            for subset_name, subset_recalls in recalls_str.items():
                logging.info(f"Recalls on eval set {subset_name}: {subset_recalls}")
        else:
            logging.info(f"Recalls on eval set {eval_dataset_name} {eval_ds}: {recalls_str}")
            writer.add_scalar(f"Val_{eval_dataset_name}/R@1", recalls[0], epoch_num)
        if eval_dataset_name == primary_eval_name:
            primary_recalls = recalls

    if primary_recalls is None:
        raise RuntimeError(f"Primary eval dataset '{primary_eval_name}' not found in eval_dataset_names")

    current_r1_r5 = r1_plus_r5(primary_recalls)
    pitts_recalls = eval_recalls.get("pitts30k", [])
    msls_recalls = eval_recalls.get("Msls_740", [])
    pitts_r1_r5 = r1_plus_r5(pitts_recalls) if len(pitts_recalls) else None
    msls_r1_r5 = r1_plus_r5(msls_recalls) if len(msls_recalls) else None
    current_pitts_r1 = recall_at(pitts_recalls, 1)
    current_msls_r1 = recall_at(msls_recalls, 1)
    has_joint_selection = len(pitts_recalls) and len(msls_recalls)
    is_best = (
        is_better_checkpoint(current_pitts_r1, current_msls_r1, best_pitts_r1, best_msls_r1)
        if has_joint_selection else
        current_r1_r5 > best_r1_r5
    )
    epoch_record = {
        "run_id": run_id,
        "epoch": epoch_num,
        "lr": args.lr,
        "current_lr": current_lr,
        "train_batch_size": args.train_batch_size,
        "freeze_te": args.freeze_te,
        "num_learnable_aggregation_tokens": args.num_learnable_aggregation_tokens,
        "training_dataset": args.training_dataset,
        "initialization_dataset": args.initialization_dataset,
        "avg_triplet_loss": float(np.mean(epoch_losses)),
        "pitts_r1": current_pitts_r1,
        "pitts_r5": recall_at(pitts_recalls, 5),
        "pitts_r10": recall_at(pitts_recalls, 10),
        "pitts_r100": recall_at(pitts_recalls, 100),
        "msls_r1": current_msls_r1,
        "msls_r5": recall_at(msls_recalls, 5),
        "msls_r10": recall_at(msls_recalls, 10),
        "msls_r100": recall_at(msls_recalls, 100),
        "primary_r1": recall_at(primary_recalls, 1),
        "primary_r5": recall_at(primary_recalls, 5),
        "primary_r1_plus_r5": current_r1_r5,
        "is_best": is_best,
        "save_dir": args.save_dir,
    }
    epoch_records.append(epoch_record)
    current_selection_key = (current_pitts_r1, current_msls_r1) if has_joint_selection else (current_r1_r5,)
    best_epoch_key = None
    if best_epoch_record is not None:
        if has_joint_selection:
            best_epoch_key = (best_epoch_record["pitts_r1"], best_epoch_record["msls_r1"])
        else:
            best_epoch_key = (best_epoch_record["primary_r1_plus_r5"],)
    if best_epoch_record is None or current_selection_key > best_epoch_key:
        best_epoch_record = epoch_record

    should_stop = False
    if is_best:
        if has_joint_selection:
            logging.info(
                f"Improved: previous best (Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f}), "
                f"current=({current_pitts_r1:.1f}, {current_msls_r1:.1f})"
            )
            best_pitts_r1 = current_pitts_r1
            best_msls_r1 = current_msls_r1
        else:
            logging.info(
                f"Improved: previous best (R@1 + R@5) = {best_r1_r5:.1f}, "
                f"current (R@1 + R@5) = {current_r1_r5:.1f}"
            )
        best_r1_r5 = current_r1_r5
        not_improved_num = 0
    else:
        not_improved_num += 1
        if has_joint_selection:
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: "
                f"best (Pitts30k R@1, MSLS R@1)=({best_pitts_r1:.1f}, {best_msls_r1:.1f}), "
                f"current=({current_pitts_r1:.1f}, {current_msls_r1:.1f})"
            )
        else:
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: "
                f"best (R@1 + R@5) = {best_r1_r5:.1f}, "
                f"current (R@1 + R@5) = {current_r1_r5:.1f}"
            )
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            should_stop = True

    state = {
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recalls": primary_recalls,
        "best_r1_r5": best_r1_r5,
        "best_pitts_r1_r5": best_pitts_r1_r5,
        "best_msls_r1_r5": best_msls_r1_r5,
        "best_pitts_r1": best_pitts_r1,
        "best_msls_r1": best_msls_r1,
        "pitts_r1": current_pitts_r1,
        "msls_r1": current_msls_r1,
        "not_improved_num": not_improved_num,
    }
    try_save_checkpoint(util.save_checkpoint, args, state, is_best, "last_model.pth")
    if pitts_r1_r5 is not None and pitts_r1_r5 > best_pitts_r1_r5:
        logging.info(
            f"New Pitts30k SOTA: previous R@1+R@5 = {best_pitts_r1_r5:.1f}, "
            f"current R@1+R@5 = {pitts_r1_r5:.1f}. Saving best_pitts30k_model.pth"
        )
        best_pitts_r1_r5 = pitts_r1_r5
        state["best_pitts_r1_r5"] = best_pitts_r1_r5
        try_save_checkpoint(util.save_checkpoint, args, state, False, "best_pitts30k_model.pth")

    if msls_r1_r5 is not None and msls_r1_r5 > best_msls_r1_r5:
        logging.info(
            f"New MSLS SOTA: previous R@1+R@5 = {best_msls_r1_r5:.1f}, "
            f"current R@1+R@5 = {msls_r1_r5:.1f}. Saving best_msls_model.pth"
        )
        best_msls_r1_r5 = msls_r1_r5
        state["best_msls_r1_r5"] = best_msls_r1_r5
        try_save_checkpoint(util.save_checkpoint, args, state, False, "best_msls_model.pth")

    if should_stop:
        break

logging.info(f"Best (R@1 + R@5): {best_r1_r5:.1f}")
logging.info(f"Best Pitts30k (R@1 + R@5): {best_pitts_r1_r5:.1f}")
logging.info(f"Best MSLS (R@1 + R@5): {best_msls_r1_r5:.1f}")
logging.info(f"Best checkpoint by (Pitts30k R@1, MSLS R@1): ({best_pitts_r1:.1f}, {best_msls_r1:.1f})")
logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

for best_name in ["best_pitts30k_model.pth", "best_msls_model.pth"]:
    best_model_path = join(args.save_dir, best_name)
    if os.path.exists(best_model_path):
        logging.info(f"Test *{best_name}* on test sets")
        best_model_state_dict = torch.load(best_model_path, map_location=args.device, weights_only=False)["model_state_dict"]
        model.load_state_dict(best_model_state_dict)
        for eval_dataset_name, eval_ds in eval_datasets.items():
            recalls, recalls_str = test.test(args, eval_ds, model, test_method=args.test_method)
            if isinstance(recalls_str, dict):
                for subset_name, subset_recalls in recalls_str.items():
                    logging.info(f"[{best_name}] Recalls on {subset_name}: {subset_recalls}")
            else:
                logging.info(f"[{best_name}] Recalls on {eval_dataset_name} {eval_ds}: {recalls_str}")
    else:
        logging.warning(f"{best_name} was not saved because that eval set did not improve or was not evaluated: {best_model_path}")

writer.close()
append_log_footer("finished")
