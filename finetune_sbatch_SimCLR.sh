#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=atmlaml
#SBATCH --cpus-per-task=1
#SBATCH --output=run-out.%j
#SBATCH --error=run-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpus

source  /p/home/jusers/shitole1/juwels/shared/varun_SSL/pytorchlightning_environment/activate.sh 

module list
python -m wandb offline

python main_linear.py \
    --dataset tbc \
    --encoder resnet18 \
    --data_dir /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/dataset_crop_imgs \
    --train_dir /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/dataset_crop_imgs \
    --val_dir /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/dataset_crop_imgs \
    --pretrained_feature_extractor /p/project/atmlaml/project_SSL_varun/solo-learn_SSL/ckpt/simclr/1tyazf8r/tbc-1tyazf8r-ep=99.ckpt \
    --max_epochs 100 \
    --gpus 0,1 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.005 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 64 \
    --num_workers 10 \
    --name simclr-tbc-finetune \
    --project tbc_h5_split \
    --entity varun-s \
    --wandb \
    --accelerator ddp