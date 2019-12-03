#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --cpus-per-task=1
#SBATCH --output=log_lm-%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=cpl

source activate /om/user/pqian/envs/py35
python main.py --cuda --emsize 256 --nhid 256 --dropout 0.3 --epochs 40 --data data/ptb --save model_small.pt
