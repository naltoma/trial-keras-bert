#!/bin/bash
#SBATCH --job-name keras-bert-normal
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv ~/SIF-images/test.sif python train.py
date
