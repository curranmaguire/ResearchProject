#!pip install torch transformers nltk scikit-learn datasets matplotlib plotly
#this code takes ~8hours to process and output. Figures will be saved in graphs
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split   
import os
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk import word_tokenize, ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import re
#code taken from this tutorial
#https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
#=====================================config
batch_size = 256
sample_size = 500

def clean_text(text):
    text = re.sub(r'\\[a-zA-Z]+(?![a-zA-Z])', '', text)
    
    # Remove specific LaTeX artifacts like @math1 and \cite{...}
    text = re.sub(r'@math\d+', '', text)  # Assuming @math1, @math2, etc.
    text = re.sub(r'@xmath\d+', '', text) 
    text = re.sub(r'\\cite\{[^}]*\}', '', text)  # Remove \cite{...}
    text = re.sub(r'&nbsp;','', text)
    text = re.sub(r'xcite','', text)
    
    # Normalize ellipses and multiple exclamation/question marks to a single instance
    text = re.sub(r'\.\.\.+', '…', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'([^\s\w,\.]|_)+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    return text


print('loading data.....')
scientific_raw = load_dataset('scientific_papers', 'arxiv', trust_remote_code=True,cache_dir='/home2/cgmj52/ResearchProject/Data/Datasets' )
journalistic_raw = load_dataset('multi_news', trust_remote_code=True , )
stories_raw = load_dataset('roneneldan/tinystories', )
gpt_dataset = load_dataset("aadityaubhat/GPT-wiki-intro",cache_dir='/home2/cgmj52/ResearchProject/Data/Datasets')

#generate samples
gpt_sample = gpt_dataset['train']['generated_intro'][:sample_size]
stories_sample = stories_raw['train']['text'][:sample_size]
journal_sample = journalistic_raw['train']['document'][:sample_size]
scientific_sample = scientific_raw['train']['abstract'][:sample_size]
sci_clean = map(clean_text, scientific_sample)
print('data loaded now processing ........')
#create an itterable so we can process each dataset
datasets = [gpt_sample, stories_sample, journal_sample, scientific_sample, sci_clean]


# Set a random seed
random_seed = 42
random.seed(random_seed)
 
# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    device = torch.device('cuda')
print(f'device is {device}')


dataset_labels = ['GPT style', 'stories', 'news reports', 'scientific', 'cleaned scientific']




'''
for i, dataset in enumerate(datasets):
    tokenized_sentences = [word_tokenize(sentence) for sentence in dataset if type(sentence) != None]

    sentence_lengths = [len(sentence) for sentence in tokenized_sentences if type(sentence) != None]

    #tokenize and get the sentence lengths
    if sentence_lengths:
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        print(f"Average sentence length for {dataset_labels[i]}: {avg_sentence_length:.2f} words")
    else:
        print(f"No sentences found in {dataset_labels[i]} dataset.")
    # Flatten the tokenized sentences into a single list of words
    words = [word for sentence in tokenized_sentences for word in sentence]

    # Create a BigramCollocationFinder object
    finder = BigramCollocationFinder.from_words(words)

    # Apply frequency filter to remove rare collocations
    finder.apply_freq_filter(3)  # Adjust the frequency threshold as needed

    # Extract the top collocations
    top_collocations = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)  # Adjust the number of top collocations as needed

    print(f"Top Collocations for {dataset_labels[i]}")
    for collocation in top_collocations:
        print(collocation)
    print('\n\n')
    
    # Specify the value of N for N-grams
    n = 3  # Change this to the desired value of N

    # Extract N-grams from the tokenized sentences
    n_grams = [list(ngrams(sentence, n)) for sentence in tokenized_sentences]

    # Flatten the list of N-grams
    flat_n_grams = [gram for sublist in n_grams for gram in sublist]

    # Count the frequency of each N-gram
    n_gram_freq = nltk.FreqDist(flat_n_grams)

    # Print the most common N-grams
    print(f"Most common {n}-grams for {dataset_labels[i]}:")
    for n_gram, freq in n_gram_freq.most_common(10):  # Adjust the number of most common N-grams as needed
        print(f"{n_gram}: {freq}")
'''



print('starting sentence embedding....')
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
#create lists to store the data and dataset_labels
word_embedding_datasets = []
embeddings_labels = []
for y, dataset in enumerate(datasets[:-1]):
    #process in batches
    for i in np.arange(0, len(dataset), batch_size):

        encoding = tokenizer.batch_encode_plus(
            dataset[i],                    # List of input texts
            padding=True,              # Pad to the maximum sequence length
            truncation=True,           # Truncate to the maximum sequence length if necessary
            return_tensors='pt',      # Return PyTorch tensors
            add_special_tokens=True    # Add special tokens CLS and SEP
        )
 
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # Generate embeddings using BERT model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state[:,-1,:].numpy()  # This contains the embeddings
            word_embedding_datasets.append(word_embeddings)
            embeddings_labels += [y]*len(word_embeddings)
    print(f'completed dataset....{dataset_labels[y]}')

word_embedding_datasets = np.concatenate(word_embedding_datasets, axis=0)


print(f'finnished embeddings with shape {word_embedding_datasets.shape}')
#print(len(word_embedding_datasets))
#max_length = max(len(seq) for seq in word_embedding_datasets)
# Pad the sequences to the maximum length
#word_embedding_datasets = np.pad(word_embedding_datasets, ((0, 0), (0, max_length - word_embedding_datasets.shape[1]), (0, 0)), mode='constant')

# Convert to a single NumPy array
print(f'starting perplexity calculations.....')

perplexity = np.arange(5,500, 50)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, perplexity=i)
    reduced = model.fit_transform(word_embedding_datasets)
    divergence.append(model.kl_divergence_)


fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
fig.write_image('graphs/perplexity_plot.png')
print('completed perplexity calculations, starting final plot....')
#takes around 4 hrs

# Plot the t-SNE results
model = TSNE(n_components=2, perplexity=150)
tsne_embeddings = model.fit_transform(word_embedding_datasets)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange']
for i, label in enumerate(dataset_labels[:-1]):
    mask = [x == i for x in embeddings_labels]
    plt.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], c=colors[i], alpha=0.8, label=label)

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Plot of Word Embeddings')
plt.legend()
plt.savefig('graphs/t-sne_plot.png')
