#!/bin/sh

#SBATCH --account=edu
#SBATCH -J mteb-energy-distance
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --ntasks-per-node=1         # Number of processes per node (should be equal to the number of GPUs per node)
#SBATCH --time=90:00:00
#SBATCH --output=output_%j.log      # Standard output log file (%j is the job ID)
#SBATCH --error=error_%j.log        # Standard error log file

module load cuda92/toolkit
module load anaconda

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
conda activate myenv39           # Activate the virtual environment
nvidia-smi

# Set PYTHONPATH to prioritize the virtual environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH

#export CUDA_VISIBLE_DEVICES=1  # Only use GPU 1

#srun python eval_dataset.py
srun python eval_dataset_subset_length.py
#srun python query_length.py
