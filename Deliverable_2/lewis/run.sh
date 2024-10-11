#!/bin/bash

#SBATCH --job-name=LEWIS_train
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=96:00:00 #a week of training
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=ALL

#SBATCH --mail-user=cgmj52@durham.ac.uk
#SBATCH --qos=long-high-prio



# Activate your virtual environment (if required)
eval "$(~/miniconda3/bin/conda shell.bash hook)"
#conda init
conda activate lewis
export PYTHONPATH=$PYTHONPATH:/home2/cgmj52/ResearchProject/Deliverable_2/lewis/fairseq
bash roberta-classifier/preprocess-roberta-classifier.sh data gpt-sci gpt sci
echo '---------------------------preprocessed'

bash roberta-classifier/train-roberta-classifier.sh data gpt-sci
echo '---------------------------trained classifier'

python roberta-classifier/convert_roberta_original_pytorch_checkpoint_to_pytorch.py --roberta_checkpoint_path data/roberta-classifier/gpt-sci/checkpoints --pytorch_dump_folder_path data/roberta-classifier-py/gpt-sci --classification_head classification_head
echo '---------------------------saved to pytorch'

bash bart-denoising/preprocess-bart-denoising.sh data gpt-sci gpt
bash bart-denoising/preprocess-bart-denoising.sh data gpt-sci sci
echo '---------------------------bart preprocessed'

bash bart-denoising/train-bart-denoising.sh data gpt-sci sci # roughly 42 hrs to reach 20k updates
bash bart-denoising/train-bart-denoising.sh data gpt-sci gpt #
echo '---------------------------finnished training bart'

python get_synthesized_data.py --d1_model_path data/bart-denoising/gpt-sci/gpt/checkpoints/checkpoint_best.pt --d2_model_path data/bart-denoising/gpt-sci/sci/checkpoints/checkpoint_best.pt --d1_file data/gpt-sci/gpt/train.txt --d2_file data/gpt-sci/sci/train.txt --out_file output/data/out.json --hf_dump data/roberta-classifier-py/gpt-sci
echo '---------------------------finnished data synthesis'

python extract_parallel_from_json.py --input_json output/data/out.json --output_path data/synth-gpt-sci
echo '---------------------------finnished extraction of parallels'
#para.d0/d1 and masks.d0/d1 in data/synth-gpt-sci

bash bart-mt/preprocess-bart-mt.sh data synth-gpt-sci gpt-sci 100000
echo '---------------------------completed bart preprocess'

bash bart-mt/train-bart-mt.sh data synth-gpt-sci gpt-sci
echo '---------------------------completed bart mt training'
python roberta-tagger/preprocess-roberta-tagger.py --path-to-parallel-data-file data/synth-gpt-sci/para.d0 --mask-threshold 6 --output-path output/data/parallel_out.json
echo '---------------------------completed bert tagger preprocess'

python roberta-tagger/train-roberta-tagger.py --path-to-parallel-data-json output/data/parallel_out.json --hf_dump data/roberta-classifier-py/gpt-sci/checkpoint.pt --save-dir data/synth-gpt-sci/roberta-tagger --epochs 20 --bsz 256 --update-freq 5
echo '---------------------------completed training tagger'

python roberta-tagger/inference-roberta-tagger.py --hf-dump data/synth-gpt-sci/roberta-tagger --path-to-parallel-data-json output/data/parallel_out.json --bsz 50 --update-freq 1 --output-path output/data
echo '---------------------------complated tagger inference'

python inference-lewis.py --input_file_path output/data/masks.txt --output_file_path output/final --bart-mt-checkpoint-path data/bart-mt/synth-gpt-sci --bart-mt-data-bin-path data/bart-mt/synth-gpt-sci/bin --hf-dump data/roberta-classifier-py/gpt-sci --target_label_index 1
#https://github.com/machelreid/lewis
echo '---------------------------finnished with inference'

conda deactivate