#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=9000
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --error="/home/<zih-login>/myjob-%A_%a.out"
#SBATCH --output="/home/<zih-login>/myjob-%A_%a.out"
#SBATCH --array=1-24
#                ^
#                |--------------------- you can use however many jobs you want. Waitinng time will increase though.

ml purge
ml release/24.04  GCC/13.2.0 OpenMPI/4.1.6 SciPy-bundle/2023.11
ml PyTorch/2.1.2
ml gym[box2d]/0.17.1
ml gym/0.26.2

# The Task-ID is an increasing integer (in this script from 1 to 24). 
# You can use it to generate combinations in this script directly, 
# or use it to index some combinations array/dict in your python code.
echo $SLURM_ARRAY_TASK_ID

srun python path/to/script.py $SLURM_ARRAY_TASK_ID`


