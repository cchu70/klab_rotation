#!/bin/bash
#SBATCH --partition=gpu                      # Partition to run in
#SBATCH --gres=gpu:1                         # GPU resources requested
#SBATCH -c 1                                 # Requested cores
#SBATCH --time=0-0:30                        # Runtime in D-HH:MM format
#SBATCH --mem=4GB                           # Requested Memory
#SBATCH -o %j.out                            # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e %j.err                            # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-user=claudia_chu@g.harvard.edu          # Email to which notifications will be sent

conda activate klab_env
python 08_simple_mlp_unsupervised_train.py --device cuda --seed 42 --batch_size 32 --lr 1e-3 --subset_fraction 0.05 --init_density 0.5 --margin 0.2 --output_dir ./outputs/08_simple_mlp_unsupervised/