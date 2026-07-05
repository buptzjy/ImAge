conda activate /media/data1/zhangjingyi/miniconda3/envs/ImAge
cd /media/data/zhangjingyi/ImAge
conda deactivate

pip install transformers -i https://mirrors.aliyun.com/pypi/simple/

# 用pitts30k验证
nohup python3 -u eval.py --eval_datasets_folder=/media/data/chenshunpeng/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/ImAge_GSV.pth > log_test_pitts30k.txt 2>&1 

# 用msls验证
nohup python3 -u eval.py --eval_datasets_folder=/media/data/chenshunpeng/datasets --eval_dataset_name=Msls_740 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/ImAge_GSV.pth > log_test_Msls_740.txt 2>&1 

# 用msls初始化
python3 -u train.py --eval_datasets_folder=/media/data/chenshunpeng/datasets  --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=5 --patience=20 --initialization_dataset= --training_dataset=gsv_cities --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_train_init—msls.txt 2>&1 &

# 用gsv_cities初始化
python3 train.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=5 --patience=20  --initialization_dataset=gsv_cities --training_dataset=gsv_cities --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_train_init—gsv_322train.txt 2>&1 &

# mix训练 无初始化
CUDA_VISIBLE_DEVICES=0,1 python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=20 --patience=20  --initialization_dataset=none  --training_dataset=LosAngeles_mix --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_trainmix_noinit.txt 2>&1 &

# 洛杉矶训练 无初始化
CUDA_VISIBLE_DEVICES=1,2,3 python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=15 --patience=20  --initialization_dataset=none  --training_dataset=LosAngeles --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_train_LAonly.txt 2>&1 &

# 洛杉矶+rainy训练 无初始化
CUDA_VISIBLE_DEVICES=1,2,3 python3 train.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=20 --patience=20  --initialization_dataset=none  --training_dataset=LArainy --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_train_LA_rainy.txt 2>&1 &

# gsv+3city训练 无初始化
CUDA_VISIBLE_DEVICES=1,2,3 python3 train.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=20 --patience=20  --initialization_dataset=none  --training_dataset=gsv_cities,3city --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_gsv_3city.txt 2>&1

# gsv+筛选后3city图训练 无初始化0402
CUDA_VISIBLE_DEVICES=0,3 python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_names pitts30k Msls_740 sped sf_xl_small tokyo247 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=20 --patience=20  --initialization_dataset=none  --training_dataset=gsv_cities --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_gsv_3city0408.txt 2>&1


# 多数据集测试
python3 eval.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets  --eval_dataset_names pitts30k Msls_740 sped sf_xl_small tokyo247 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/media/data1/zhangjingyi/ImAge/logs/default/2026-04-09_17-13-54/best_model.pth

# 0408 gsv+3city困难正样本训练 无初始化
nohup python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_names pitts30k Msls_740 sped sf_xl_small tokyo247 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00002 --epochs_num=20 --patience=20  --initialization_dataset=none  --training_dataset=gsv_cities --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_test0527_tmp.txt 2>&1

# Test 0422
python3 eval.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets  --eval_dataset_names pitts30k Msls_740 sped sf_xl_small tokyo247 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/media/data/zhangjingyi/ImAge/logs/default/2026-04-21_16-13-47/best_model.pth
# test 0424
python3 eval.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets  --eval_dataset_names pitts30k Msls_740 sped sf_xl_small tokyo247 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/media/data1/chenshunpeng1/project/VPR2/ImAge-main/logs/default/2026-03-06_01-10-21/best_model.pth

# 0528 训练
python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00003 --epochs_num=20 --patience=20 --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --initialization_dataset=none  --training_dataset=unified_dataset --foundation_model_path=/media/data/zhangjingyi/ImAge/module/ImAge_Merged.pth > log_308train_init—unified_0529.txt 2>&1 &

# 0530 训练（lufeng权重）
nohup python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --train_image_size 252 252 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00002 --epochs_num=20 --patience=20 --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --initialization_dataset=gsv_cities --training_dataset=unified_dataset --foundation_model_path=/media/data/zhangjingyi/ImAge/module/ImAge_Merged.pth > log_train_init—unified252_lufeng.txt 2>&1 &

# 0605训练（vitb权重，266分辨率，pitts sota）
CUDA_VISIBLE_DEVICES=2 nohup python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --train_image_size 238 238 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00002 --epochs_num=20 --patience=20 --initialization_dataset=unified_dataset --training_dataset=unified_dataset --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_train_随便跑_06012.txt 2>&1 &

# 0616训练 238单卡 mixGSV训练
CUDA_VISIBLE_DEVICES=2 nohup python3 train_baseline.py --eval_datasets_folder=/media/data1/chenshunpeng1/datasets --eval_dataset_name=pitts30k --train_image_size 252 252 --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00002 --epochs_num=20 --patience=20 --initialization_dataset=unified_dataset --training_dataset=/data_nvme/zhangjingyi/ReflectCities/mixed_GSV --foundation_model_path=/media/data1/chenshunpeng1/project/VPR2/ImAge/model/dinov2_vitb14_reg4_pretrain.pth > log_train_gsvreflect_0617.txt 2>&1 &
