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

source /n/app/miniconda3/23.1.0/etc/profile.d/conda.sh # https://stackoverflow.com/questions/61915607/commandnotfounderror-your-shell-has-not-been-properly-configured-to-use-conda
conda activate klab_env

python3 src/simple_mlp_unsupervised_train.py \
    --subset_fraction 0.05 \
    --num_training_iter 50 \
    --prediction_act_type linear \
    --margin 5 \
    --use_grow_prune_prob False \
    --output_dir outputs/08/ \
    --job_id $SLURM_JOB_ID \
