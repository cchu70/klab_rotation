#!/bin/bash

#SBATCH -c 1
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 5G
#SBATCH -o 08_simple_mlp_unsupervised_train_%j.out
#SBATCH -e 08_simple_mlp_unsupervised_train_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=claudia_chu@g.harvard.edu
#SBATCH -x compute-g-16-175,compute-g-16-177,compute-g-16-194

module load gcc/9.2.0
module load python/3.9.14
source ~/klab_venv/bin/activate

#conda activate klab_env
python 08_simple_mlp_unsupervised_train.py --device cuda --seed 42 --batch_size 32 --lr 1e-3 --subset_fraction 0.05 --init_density 0.5 --margin 0.2 --output_dir ./outputs/08_simple_mlp_unsupervised/

