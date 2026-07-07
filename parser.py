
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=120,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.00005, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--epochs_num", type=int, default=20,
                        help="number of epochs to train for")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Inference parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["dinov2", "vit", "clip"], help="_")
    parser.add_argument("--aggregator", type=str, default= None,
                        choices=["netvlad", "salad", "boq", None])
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(0, 11)))
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 11)))
    parser.add_argument("--num_learnable_aggregation_tokens", type=int, default=8)
    parser.add_argument("--num_register_tokens", type=int, default=4,
                        help="Number of DINOv2 register tokens. Use 0 for SALAD/BoQ checkpoints trained without registers.")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--initialization_dataset", type=str, default="msls_train",
                        help="sample place images to initialize learnable aggregation tokens; can be a known dataset name or a dataset root path")
    parser.add_argument("--training_dataset", type=str, default="gsv_cities",
                        help="Dataset for model training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[322, 322], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--train_image_size', type=int, default=[224, 224], nargs=2,
                        help="Training image size (HxW) for get_GSVCities.")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument(
        "--efficient_ram_testing",
        action="store_true",
        help=("Use exact float32 L2 retrieval with a dynamically chunked database. "
              "This greatly reduces RAM usage at the cost of additional search overhead."),
    )
    parser.add_argument(
        "--database_chunk_size",
        type=int,
        default=8192,
        help="Database chunk size for exact low-RAM retrieval.",
    )
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--eval_datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k",
                        help="Primary dataset used to select the best checkpoint")
    parser.add_argument("--eval_dataset_names", type=str, nargs="+",
                        default=["Msls_740", "pitts30k", "sped", "amstertime", "tokyo247", "nordland"],
                        help="List of datasets to evaluate after each epoch")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--fast_debug_batches", action="store_true",
                        help="If set, run only 3 batches per epoch before evaluation")
    parser.add_argument("--save_epoch_checkpoints", action="store_true",
                        help="If set, save a full checkpoint for every epoch. This uses a lot of disk space.")
    args = parser.parse_args()
    
    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    return args
