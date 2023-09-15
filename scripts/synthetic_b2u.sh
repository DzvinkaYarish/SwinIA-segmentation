#!/bin/bash

#SBATCH -J gauss25
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 40:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1

module load any/python/3.8.3-conda

conda activate n2s_env

cd noise2same || return

python train.py +backbone=unet_b2u +denoiser=blind2unblind +experiment=synthetic project=synthetic \
                dataset.noise_param=25 dataset.noise_type=gaussian \
                dataset.mean=[0,0,0] dataset.std=[255,255,255]