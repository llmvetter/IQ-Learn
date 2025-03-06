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
source /home/h6/leve469a/IQ-Learn/.venv/bin/activate
srun python /home/h6/leve469a/IQ-Learn/hp_search.py $SLURM_ARRAY_TASK_ID`


