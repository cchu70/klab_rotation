#!/bin/bash

#SBATCH -c 1
#SBATCH -t 0-1:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 5G
#SBATCH -o outputs/sbatch/%j/stdout
#SBATCH -e outputs/sbatch/%j/sterr
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=claudia_chu@g.harvard.edu
#SBATCH -x compute-g-16-175,compute-g-16-177,compute-g-16-194

source /n/app/miniconda3/23.1.0/etc/profile.d/conda.sh
conda activate klab_env

 # --use_grow_prune_prob False 
python3 /home/clc926/Desktop/klab_rotation/src/simple_mlp_SUPERVISED_train.py --subset_fraction 0.1 \
    --num_training_iter 500 \
    --output_dir outputs/11/ \
    --job_id $SLURM_JOB_ID \
    --desc no_prune_full