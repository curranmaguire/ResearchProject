#!/bin/bash

#SBATCH --job-name=ling_train
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00 #a week of training
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=ALL

#SBATCH --mail-user=cgmj52@durham.ac.uk
#SBATCH --qos=short

input_file="/home2/cgmj52/ResearchProject/Deliverable_2/lewis/data/gpt-sci/gpt/test.txt"
source .env/bin/activate

#python /home2/cgmj52/ResearchProject/Deliverable_2/linguistic-style-transfer-pytorch/linguistic_style_transfer_pytorch/utils/preprocess.py


#python /home2/cgmj52/ResearchProject/Deliverable_2/linguistic-style-transfer-pytorch/linguistic_style_transfer_pytorch/utils/train_w2v.py
#python /home2/cgmj52/ResearchProject/Deliverable_2/linguistic-style-transfer-pytorch/linguistic_style_transfer_pytorch/utils/vocab.py

#python train.py #This takes ~48hrs
# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file not found: $input_file"
    exit 1
fi
 #Loop through each line of the input file
while IFS= read -r line; do
    # Pass the line to the Python script and redirect the output to generate_text.txt
    python generate.py "$line" >> generate_text.txt
done < "$input_file"


