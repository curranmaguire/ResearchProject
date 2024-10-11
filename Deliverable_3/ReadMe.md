# Deliverable 3: Discourse Coherence Generation (DCG) Model

This project implements a Discourse Coherence Generation (DCG) model for text style transfer. It includes scripts for data preprocessing, model training, and evaluation.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Running the Project](#running-the-project)

## Project Structure

- `DCG_classifier_train.py`: Script for training the classifier model
- `DCG_bart_train.py`: Script for training the BART model
- `DCG_model_implementation.ipynb`: Jupyter notebook for model implementation details
- `Evaluate_models.py`: Script for evaluating the trained models
- `run.sh`: Bash script to run the entire pipeline

## Setup

1. Ensure you have Conda installed on your system.
2. Create a new Conda environment:
   ```
   conda create -n dcg_env python=3.9
   conda activate dcg_env
   ```
3. Install the required packages:
   ```
   pip install torch transformers datasets sklearn nltk evaluate lmppl
   ```

## Data Preprocessing

The data preprocessing is handled in the `DCG_model_implementation.ipynb` notebook. It includes:

1. Loading the dataset (scientific papers from arXiv)
2. Cleaning and tokenizing the text
3. Creating train and test splits
4. Preparing the data for the BERT model

## Model Training

### Classifier Training
Run `DCG_classifier_train.py` to train the classifier model. This script:
- Uses a BERT-based model for sequence classification
- Trains on coherent and incoherent sentence pairs
- Saves the best model based on evaluation metrics

### BART Model Training
Run `DCG_bart_train.py` to train the BART model. This script:
- Fine-tunes a BART model for conditional generation
- Uses the trained classifier to guide the training process
- Saves the best model based on validation loss

## Evaluation

The `Evaluate_models.py` script provides comprehensive evaluation of the trained models:

- Perplexity (PPL) scoring using GPT-2
- BERT Score for semantic similarity
- Style accuracy using the trained classifier
- BLEU score for content preservation

## Running the Project

Use the `run.sh` script to execute the entire pipeline:

1. Ensure you have SLURM installed on your system.
2. Submit the job using:
   ```
   sbatch run.sh
   ```

This script will:
1. Train the classifier model
2. Train the BART model
3. Run inference and evaluation

Note: The script is configured for a SLURM-based cluster. Adjust the SLURM parameters as needed for your system.

## Notes

- The project uses PyTorch and Hugging Face's Transformers library.
- GPU acceleration is recommended for faster training.
- Adjust hyperparameters in the respective scripts as needed for your specific use case.

For any issues or questions, please open an issue in the project repository.