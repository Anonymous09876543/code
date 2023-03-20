#!/bin/bash
#SBATCH -e result/exp-his.err
#SBATCH -o result/exp-his.out
#SBATCH -J prefix-tuning
#SBATCH --partition=si
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=999:00:00
python train.py --cfg configs/models/regression_metric.yaml
