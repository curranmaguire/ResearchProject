#!/bin/bash

#SBATCH --job-name=myPythonJob   # Job name
#SBATCH --partition=ug-gpu-small          # Partition name
#SBATCH --gres=gpu:pascal   # Request GPU resource
#SBATCH --mem=4G                 # Memory total in MB (for all cores)
#SBATCH --qos=short
#SBATCH --time=01:00:00          # Time limit hrs:min:sec
#SBATCH --output=myPythonJob.log # Standard output and error log

# Load the module for CUDA and cuDNN in case they are required
#module load cuda/10.0
#module load cudnn/7.6-cuda-10.0

# Activate your Conda environment
# conda activate mykernel

source /home2/cgmj52/myjupyterenv/bin/activate
# Run your Python script
python3 /home2/cgmj52/ResearchProject/Data/classifier_training.py
deactivate
# Deactivate Conda environment
# conda deactivate