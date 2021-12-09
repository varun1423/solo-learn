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

source  ./../shared/varun_SSL/pytorchlightning_environment/activate.sh 

module list

python /p/home/jusers/shitole1/juwels/solo-learn_SSL/main_pretrain.py \
    --dataset tbc \
    --encoder resnet18 \
    --data_dir /p/home/jusers/shitole1/juwels/shared/varun_SSL/solo-learn/datasets/data_public_leaderboard_phase/  \
    --train_dir /p/home/jusers/shitole1/juwels/shared/varun_SSL/solo-learn/datasets/data_public_leaderboard_phase//train/ \
    --checkpoint_dir /p/home/jusers/shitole1/juwels/solo-learn_SSL/ckpt/ \
    --max_epochs 400 \
    --gpus 0,1,2,3 \
    --accelerator ddp \
    --sync_batchnorm \
    --num_workers 4 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name barlow\
    --entity varun-s \
    --project SSL_Barlow \
    --save_checkpoint \
    --scale_loss 0.1 \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --wandb
