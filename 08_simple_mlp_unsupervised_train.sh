#!/bin/bash

#SBATCH -c 1
#SBATCH -t 0-1:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 5G
#SBATCH -o outputs/08/%j/stdout
#SBATCH -e outputs/08/%j/sterr
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=claudia_chu@g.harvard.edu
#SBATCH -x compute-g-16-175,compute-g-16-177,compute-g-16-194

module load miniconda3/23.1.0 gcc/9.2.0 cuda/12.1
source ~/n/app/miniconda3/23.1.0
conda activate klab_env

python 08_simple_mlp_unsupervised_train.py 
    --num_training_iter 50 \
    --prediction_act_type linear \
    --margin 5 \
    --use_grow_prune_prob False \
    --output_dir outputs/08/%j/ \
