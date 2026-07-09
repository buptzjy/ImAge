#!/usr/bin/env bash
set -e

cd /media/data/zhangjingyi/ImAge/finetuning
PYTHON_BIN=/media/data1/zhangjingyi/miniconda3/envs/IC-Light/bin/python

echo "[$(date "+%F %T")] start ImAge tmp_group=pitts on GPU1,2"
CUDA_VISIBLE_DEVICES=1,2 PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_mixed_resume.py \
  --aggregator image \
  --resume_author /media/data/zhangjingyi/ImAge/module/ImAge_GSV.pth \
  --training_dataset /data_nvme/zhangjingyi/Gsv_reflect/mixgsv \
  --training_subsets default tmp \
  --tmp_group pitts \
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets \
  --eval_dataset_names Msls_740 pitts30k \
  --final_eval_dataset_names sped amstertime tokyo247 nordland SVOX \
  --train_batch_size 120 \
  --train_image_size 322 322 \
  --resize 322 322 \
  --lr 5e-6 \
  --epochs_num 20 \
  --warmup_epochs 5 \
  --real_to_synthetic 8 \
  --freeze_te 8 \
  --patience 5 \
  --save_dir /media/data/zhangjingyi/ImAge/logs/image_mixed_resume_sfxl_pitts_fix_0706_gpu12

echo "[$(date "+%F %T")] start ImAge tmp_group=msls on GPU1,2"
CUDA_VISIBLE_DEVICES=1,2 PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_mixed_resume.py \
  --aggregator image \
  --resume_author /media/data/zhangjingyi/ImAge/module/ImAge_GSV.pth \
  --training_dataset /data_nvme/zhangjingyi/Gsv_reflect/mixgsv \
  --training_subsets default tmp \
  --tmp_group msls \
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets \
  --eval_dataset_names Msls_740 pitts30k \
  --final_eval_dataset_names sped amstertime tokyo247 nordland SVOX \
  --train_batch_size 120 \
  --train_image_size 322 322 \
  --resize 322 322 \
  --lr 5e-6 \
  --epochs_num 20 \
  --warmup_epochs 5 \
  --real_to_synthetic 8 \
  --freeze_te 8 \
  --patience 5 \
  --save_dir /media/data/zhangjingyi/ImAge/logs/image_mixed_resume_sfxl_msls_fix_0706_gpu12

echo "[$(date "+%F %T")] done"
