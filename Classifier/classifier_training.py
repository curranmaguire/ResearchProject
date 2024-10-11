import pandas as pd
import numpy as np
from datasets import load_from_disk, Dataset
import os
import torch

from pytorch_transformers import BertTokenizer, BertModel, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = os.getcwd() + '/Datasets/combined_data'
train = load_from_disk(os.path.join(data_path,'train')).with_format('torch' ).shuffle(seed=42).select(range(10000))
test = load_from_disk(os.path.join(data_path, 'test')).with_format('torch').shuffle(seed=42).select(range(10000))
validation = load_from_disk(os.path.join(data_path, 'validation')).with_format('torch').shuffle(seed=42).select(range(10000))
from torch.utils.data import DataLoader


print(f'loaded the data. Moving onto tokenizing and cleaning.')

import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def remove_latex_markup(text):
    # Remove inline math wrapped in $...$
    text = re.sub(r'\$.*?\$', '', text)
    
    # Remove display math wrapped in \[...\]
    text = re.sub(r'\\\[.*?\\\]', '', text)
    
    # Remove simple LaTeX commands like \command{arg}
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
    
    # Remove custom @xmath and @xcite commands from the provided example
    text = re.sub(r'@\w+', '', text)
    
    # Remove remaining braces after previous replacements
    text = re.sub(r'[\{\}\[\]]', '', text)
    
    return text

def clean_text(text):
    """Apply text cleaning and normalization steps."""
    # Normalize excessive punctuation, remove numbers, and LaTeX
    text = re.sub(r"\.\.\.+", ".", text)
    text = re.sub(r"\d+(\.\d+)?", "", text)
    text = remove_latex_markup(text)
    # Further custom cleaning steps can be added here
    return text

def preprocess_and_tokenize(batch):
    """Clean and tokenize texts using the BERT tokenizer."""
    # Apply custom text cleaning
    batch["text"] = [clean_text(text) for text in batch["text"]]
    tokenized_inputs = tokenizer(batch["text"], padding=True, max_length=512, truncation=True, return_tensors="pt")
    batch.update(tokenized_inputs)
    return batch


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

encoded_train = train.map(preprocess_and_tokenize, batched=True).shuffle()
encoded_eval = validation.map(preprocess_and_tokenize, batched=True).shuffle()
encoded_test = test.map(preprocess_and_tokenize, batched=True).shuffle()

train_dataloader = DataLoader(encoded_train, batch_size=16)
eval_dataloader = DataLoader(encoded_eval, batch_size=16)
print(f'Encoded the Data. Moving onto training')
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "BLOG", 1: "SCIENTIFIC", 2:"JOURNALISTIC", 3:"NARRATIVE"}
label2id = {"BLOG": 0, "SCIENTIFIC": 1, "JOURNALISTIC": 2, "NARRATIVE":3}

from transformers import  TrainingArguments, AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id
)

def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id
    )
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }
training_args = TrainingArguments(
    output_dir="Bert_classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
print('Finnished initial training. Now hyperparameter tuning')
train_dataset = encoded_train.shard(index=1, num_shards=10) 

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=encoded_eval,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
)
best_trials = trainer.hyperparameter_search(
    direction=["maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    
)
print('finnished Hyper parameter tuning. Moving onto final training with the best params.')
for n, v in best_trials.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

print('completed training now the best model has been trained')