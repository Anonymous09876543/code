#!/bin/bash
#SBATCH -e test/exp-his.err
#SBATCH -o test/exp-his.out
#SBATCH -J test
#SBATCH --partition=si
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=999:00:00
python score.py