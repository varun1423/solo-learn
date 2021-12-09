#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=atmlaml
#SBATCH --cpus-per-task=1
#SBATCH --output=run-out.%j
#SBATCH --error=run-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpus

source  /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh 

module list

python -m wandb offline

python main_pretrain.py \
    --dataset tbc \
    --encoder resnet18 \
    --data_dir /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/dataset_crop_imgs \
    --train_dir /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/dataset_crop_imgs \
    --checkpoint_dir /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/ckpt/ \
    --max_epochs 100 \
    --gpus 0,1,2,3 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 32 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 256 \
    --num_crops_per_aug 1 1 \
    --name tbc \
    --project tbc_crop \
    --entity varun-s \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 256 \
    --accelerator ddp \
    --debug_augmentations \
    --wandb 