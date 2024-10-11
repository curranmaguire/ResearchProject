import lmppl
from evaluate import load
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from typing import List


scorer = lmppl.LM('gpt2')

with open('LEWIS.txt') as f:
    output = f.readline()
ppl = scorer.get_perplexity(output)
print('ppl score of the test data is:\n')
print(list(zip(output, ppl)))
average_ppl = sum(ppl) / len(ppl)
print(f'average PPL score is {average_ppl}')

with open('LEWIS_input.txt') as f:
    inputs = f.readlines()

#s-bleu performs a BLEU score with the source as the reference and TST text as the hypothesis.
s_bleu = corpus_bleu(inputs, output)
print(f'source bleu score is {s_bleu}')

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

model_path = "/home2/cgmj52/ResearchProject/Classifier/Bert_classifier/checkpoint-2500"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def evaluate_style_accuracy(texts, model, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_samples = len(texts)
    correct_predictions = 0

    for output in texts:
        inputs = tokenizer(output, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        if predicted_label == 1:  # Assuming label 1 represents the desired style
            correct_predictions += 1

    style_accuracy = correct_predictions / total_samples
    return style_accuracy


style_accuracy = evaluate_style_accuracy(output, model, tokenizer)
print(f"Style accuracy: {style_accuracy:.4f}")



