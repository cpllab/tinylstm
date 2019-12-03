#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --cpus-per-task=1
#SBATCH --output=log_eval-%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=cpl

source activate /om/user/pqian/envs/py35

python eval.py --checkpoint model_small.pt --eval_data stimuli_items/input_test.raw --fpath stimuli_items/surprisals_test.txt