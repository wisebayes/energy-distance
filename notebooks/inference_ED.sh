#!/bin/sh

#SBATCH --account=edu
#SBATCH -J mteb-energy-distance
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:2               # Number of GPUs per node
#SBATCH --cpus-per-task=8           # 8 CPU cores per task
#SBATCH --mem-per-cpu=16gb	# The memory the job will use per cpu core.
#SBATCH --ntasks-per-node=1         # Number of processes per node (should be equal to the number of GPUs per node)
#SBATCH --time=10:00:00
#SBATCH --output=output_%j.log      # Standard output log file (%j is the job ID)
#SBATCH --error=error_%j.log        # Standard error log file

module purge
module load gcc/14.1.0
module load cuda

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
source /insomnia001/depts/edu/COMSE6998/ck3255/anaconda3/etc/profile.d/conda.sh
conda activate myenv39           # Activate the virtual environment
nvidia-smi

# Set PYTHONPATH to prioritize the virtual environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH

#export CUDA_VISIBLE_DEVICES=1,2  # Only use GPU 1

srun python eval_dataset.py
# srun python /insomnia001/depts/edu/COMSE6998/ck3255/energy-distance/notebooks/eval_dataset_subset_length.py
#srun python query_length.py
