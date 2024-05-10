# ResearchProject

## Data preprocessing
run the *format_data.ipynb* jupyter notebook in the Data directory. Once this has completed you should have multiple files labelled .train, .dev, .test, .whole and a corpus of each item. 

### moving data to each model

go to Deliverable_2/delete_retrieve_generate/data/GPT-BLOG
copy
- a corpus file and rename it gpt_sci.corpus
- a .whole file and rename it reference.0 or 1 where 0 refers to the source style and 1 the target
- a .dev, .train, .test file and rename them sentiment.<type>.0 or 1

go to Deliverable_2/lewis/data/gpt-sci
copy .train, .test, .dev into gpt and sci where gpt is the source style and sci is the target rename these train.txt, test.txt and valid.txt respectively

go to Deliverable_2/linguistic-style-transfer-pytorch/linguistic-style-transfer-pytorch/data/raw
move the .whole of the respective styles into there. You will have to change the name to gpt.whole and scientific.whole to prevent changing the config.py file.

## Running the models
follow the ReadMe's in each models directory to setup the Venv for the TST models. For Lewis you will get some errors around fairseq if you follow it directly I recommend cloning fairseq directly into the lewis repository and then continuing to set it up as an exicutable to stop getting the .examples does not exist error. 

Then cd into each directoy and use the run.sh file provided. you will have to run lewis on ampere as it generates OOM errors on smaller GPUs 

### some things to note
- Lewis takes about two weeks to train fully. However this is not done in one go and you can comment out parts of the BASH script to allow the model to run for longer.
- Lewis uses Fairseq to train itself which is a very buggy directory and I have had a lot of problems with implementation issues that seem to be specific to fairseq.
- DeleteRetrieveGenerate also takes about a week to train however inference starts quite early on in the process so you can evaluate the models outputs. 

### evaluation
use the evaluation scripts in the deliverables directory.

