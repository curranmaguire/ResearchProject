#!/bin/bash

#SBATCH --job-name=data
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgmj52@durham.ac.uk

conda activate lewis

#pip install torch transformers nltk scikit-learn datasets matplotlib plotly
#pip install -U kaleido
python data_analysis.py
