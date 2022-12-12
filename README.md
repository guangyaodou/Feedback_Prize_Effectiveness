# Feedback_Prize_Effectiveness

A model which classifies argumentative elements such as Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, and
Concluding Statement as "effective," "adequate," or "ineffective." based on essays written by U.S. students in grades
6-12. The Kaggle competition can be found [here](https://www.kaggle.com/competitions/feedback-prize-effectiveness/overview).

### Installation and Dependencies:

We recommend using conda environment to install dependencies of this library first. Please install (or load) conda and
then proceed with the following commands:

```
conda create -n dl_project python=3.9
conda activate dl_project
conda install pytorch torchvision torchaudio -c pytorch
conda install -c pytorch torchtext
conda install matplotlib
pip install pyyaml
pip install datasets
pip install transformers
```

Sometimes, there might be some errors when using torchtext such as

```
ModuleNotFoundError: No module named 'torchtext.legacy'
```

In this case, try downgrade your torchtext

```
pip install torchtext==0.10.0
```

Next, please install spacy since it is our default tokenizer

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```

or, if you are using ARM/M1, run this:

```
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm
```

### Code Hierarchy Table

- Inside the code folder

| File                                            | Description                                                                                             |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [review_data.ipynb](code/review_data.ipynb)     | Plot several bar chart to better understand the train data                                           |
| [preprocessing.py](code/preprocessing.py)       | Prepare train, valid, and test data for classic neural network RNN, LSTM, and GRU                                           |
| [models.py](code/models.py)                     | Defines the RNN, LSTM, and GRU                                                                      |
| [bert_preprocess.py](code/bert_preprocess.py)   | Prepare train, valid, and test data for bert and debert model including toknizering data, concating discourse type with discouse_text etc.                                           |                         
| [config.yml](code/config.yml)                   | Sets the hyperparameter for loading data, initializing models, and training.                            |
| [train.py](code/train.py)                       | Defines functions that train the model, plot loss/accuracy for train/valid datasets, and make inference |
| [run.py](code/run.py)                           | Trains the model, plot loss and accuracy                                                                |

### Model's performance

| Model  | Bidirectional | Last Hidden | Loss on train | Loss on validation | 
|--------|---------------|-------------|---------------|--------------------|
| RNN    | True          | True        | 0.825         | 0.883              |
| RNN    | True          | False       | 0.860         | 0.888              |
| RNN    | False         | True        | 0.936         | 0.929              |
| RNN    | False         | False       | 0.873         | 0.892              |
| LSTM   | True          | True        | 0.849         | 0.876              |
| LSTM   | True          | False       | 0.842         | 0.880              |
| LSTM   | False         | True        | 0.843         | 0.885              |
| LSTM   | False         | False       | 0.836         | 0.888              |
| GRU    | True          | True        | 0.830         | 0.879              |
| GRU    | True          | False       | 0.833         | 0.878              |
| GRU    | False         | True        | 0.848         | 0.883              |
| GRU    | False         | False       | 0.854         | 0.884              |
| BERT   | N/A           | N/A         | N/A           | 0.733              |
| DeBert | N/A           | N/A         | N/A           | 0.730              |


## Report

Here is the link of the
project [report](https://docs.google.com/document/d/1eNK_QezpReO-WvoyJmVDHdaUQdjPXsflrAp-O2gcidA/edit?usp=sharing).
