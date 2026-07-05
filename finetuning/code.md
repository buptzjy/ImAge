cd /media/data/zhangjingyi/ImAge

# SALAD
CUDA_VISIBLE_DEVICES=1 nohup python -u finetuning/train_mixed_resume.py \
  --aggregator salad \
  --resume_author /media/data/zhangjingyi/ImAge/module/dino_salad.ckpt \
  --training_dataset /data_nvme/zhangjingyi/Gsv_reflect/mixgsv \
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets \
  --train_batch_size 60 \
  --lr 5e-6 \
  --epochs_num 5 \
  --warmup_epochs 0 \
  --real_to_synthetic 8 \
  --save_dir salad_mixed_resume \
  > log_salad_mixGSV_resume_0626.txt 2>&1 &


# BOQ
cd /media/data/zhangjingyi/ImAge
conda activate ImAge

CUDA_VISIBLE_DEVICES=0,1 nohup python -u finetuning/train_mixed_resume.py \
  --aggregator boq \
  --resume_author /media/data/zhangjingyi/ImAge/module/dinov2_12288.pth \
  --training_dataset /data_nvme/zhangjingyi/ReflectCities/mixed_GSV \
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets \
  --eval_dataset_names Msls_740 pitts30k \
  --train_batch_size 120 \
  --train_image_size 322 322 \
  --resize 322 322 \
  --lr 5e-6 \
  --epochs_num 40 \
  --warmup_epochs 0 \
  --real_to_synthetic 8 \
  --freeze_te 10 \
  --save_dir boq_mixed_resume \
  > log_boq_mixgsv322_0703.txt 2>&1 &

# ImAge自己
cd /media/data/zhangjingyi/ImAge

CUDA_VISIBLE_DEVICES=2,3 nohup python -u finetuning/train_mixed_resume.py \
  --aggregator image \
  --resume_author /media/data/zhangjingyi/ImAge/module/ImAge_GSV.pth \
  --training_dataset /data_nvme/zhangjingyi/Gsv_reflect/mixgsv \
  --training_subsets default tmp \
  --eval_datasets_folder /media/data1/chenshunpeng1/datasets \
  --eval_dataset_names Msls_740 pitts30k \
  --freeze_te 8 \
  --num_learnable_aggregation_tokens 8 \
  --train_batch_size 120 \
  --train_image_size 322 322 \
  --resize 322 322 \
  --lr 5e-6 \
  --epochs_num 40 \
  --warmup_epochs 5 \
  --real_to_synthetic 8 \
  --patience 5 \
  --save_dir image_mixed_resume \
  > log_Image_mixGSV_finetuning_0704.txt 2>&1 &