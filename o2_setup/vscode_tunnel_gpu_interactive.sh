#!/bin/bash

# from login node
module load miniconda3/23.1.0 gcc/9.2.0 cuda/12.1
echo "loaded conda, gcc, and cuda"

#SBATCH -c 3
#SBATCH -t 4:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 5G
#SBATCH --job-name="vscodetunnel"
#SBATCH -x compute-g-16-177,compute-g-16-175
echo "launding gpu job"
sleep 4h

