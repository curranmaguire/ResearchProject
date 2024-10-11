# ResearchProject

This project focuses on text style transfer using various models. Below are instructions for data preprocessing, model setup, and evaluation.

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Data Distribution](#data-distribution)
3. [Model Setup](#model-setup)
   - [Delete-Retrieve-Generate](#delete-retrieve-generate)
   - [Lewis](#lewis)
   - [Linguistic Style Transfer Pytorch](#linguistic-style-transfer-pytorch)
4. [Running the Models](#running-the-models)
5. [Evaluation](#evaluation)
6. [Notes and Considerations](#notes-and-considerations)

## Data Preprocessing

1. Navigate to the `Data` directory.
2. Run the `format_data.ipynb` Jupyter notebook.
3. After completion, you should have multiple files with extensions:
   - `.train`
   - `.dev`
   - `.test`
   - `.whole`
   - A corpus file for each item

## Data Distribution

### Delete-Retrieve-Generate Model
Location: `Deliverable_2/delete_retrieve_generate/data/GPT-BLOG`

1. Copy and rename the corpus file to `gpt_sci.corpus`
2. Copy a `.whole` file and rename it to `reference.0` or `reference.1` (0 for source style, 1 for target)
3. Copy `.dev`, `.train`, `.test` files and rename them to `sentiment.0` or `sentiment.1`

### Lewis Model
Location: `Deliverable_2/lewis/data/gpt-sci`

1. Copy `.train`, `.test`, `.dev` files into `gpt` and `sci` directories
   - `gpt` is the source style
   - `sci` is the target style
2. Rename these files to `train.txt`, `test.txt`, and `valid.txt` respectively

### Linguistic Style Transfer Pytorch Model
Location: `Deliverable_2/linguistic-style-transfer-pytorch/linguistic-style-transfer-pytorch/data/raw`

1. Move the `.whole` files of respective styles into this directory
2. Rename them to `gpt.whole` and `scientific.whole`

## Model Setup

Follow the README instructions in each model's directory to set up the virtual environment for the TST models.

### Lewis Model Setup Note
You may encounter errors related to Fairseq. To resolve:
1. Clone Fairseq directly into the Lewis repository
2. Set it up as an executable to resolve the ".examples does not exist" error

## Running the Models

1. Navigate to each model's directory
2. Use the provided `run.sh` file

Note: Run Lewis on Ampere GPUs to avoid OOM errors on smaller GPUs.

## Evaluation

Use the evaluation scripts located in the `deliverables` directory.

## Notes and Considerations

1. **Lewis Model:**
   - Full training takes approximately two weeks
   - Training is not continuous; you can comment out parts of the BASH script for longer runs
   - Uses Fairseq, which may present implementation challenges

2. **Delete-Retrieve-Generate Model:**
   - Training takes about a week
   - Inference begins early in the process, allowing for early model output evaluation

For any issues or questions, please refer to the individual model documentation or open an issue in the project repository.
