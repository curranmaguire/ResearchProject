#!/bin/bash

#SBATCH --job-name=train_delete_retrieve
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=cgmj52@durham.ac.uk



# Activate your virtual environment (if required)
source env/bin/activate
#rm vocab.txt
#rm ngram_attribute_vocab.txt
#python tools/make_vocab.py /home2/cgmj52/ResearchProject/Deliverable_2/delete_retrieve_generate/data/GPT-BLOG/gpt_sci.corpus 60000 > vocab.txt
#echo 'vocab created'
#python tools/make_ngram_attribute_vocab.py vocab.txt /home2/cgmj52/ResearchProject/Deliverable_2/delete_retrieve_generate/data/GPT-BLOG/reference.0 /home2/cgmj52/ResearchProject/Deliverable_2/delete_retrieve_generate/data/GPT-BLOG/reference.1 0.7 > ngram_attribute_vocab.txt
echo 'starting training'
python train.py --config yelp_config.json --bleu #roughly 6hrs to train and 3:20 hrs to validate

deactivate