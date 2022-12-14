import os

import numpy as np
import pandas as pd
import torch
from torchtext.legacy import data


def load_data(BATCH_SIZE=30,
              split_ratio=0.8,
              vectors="glove.6B.100d",
              data_information=False,
              ):
    """
    This function loads our datasets which are wrapped by Torchtext library
    :param BATCH_SIZE: batch size (default: 30)
    :param vectors: pre-trained word embeddings we want to use (default: "glove.6B.100d")
    :param data_information: True if you want to view information of training datasets (default: False)
    :return: train_iterator, valid_iterator, test_iterator, TEXT, LABEL
    """
    current_path = os.getcwd()
    data_dir = '/../data/'
    train_data_path = os.path.join(current_path + data_dir, "train.csv")
    test_data_path = os.path.join(current_path + data_dir, "test.csv")

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Change String type Discourse type into int number
    train_df["discourse_effectiveness"] = train_df["discourse_effectiveness"].replace(
        ['Adequate', 'Effective', 'Ineffective'], [1, 2, 0])

    # Use separater to concat the discourse type and discourse text
    sep = "[SEP]"
    train_df['discourse_text'] = train_df["discourse_type"] + \
                                 " " + sep + " " + train_df["discourse_text"]
    test_df['discourse_text'] = test_df["discourse_type"] + \
                                " " + sep + " " + test_df["discourse_text"]

    # drop discourse type and rename effectiveness as label
    train_df = train_df.drop(columns=["discourse_type"])
    train_df = train_df.rename(columns={"discourse_effectiveness": "label"})

    test_df = test_df.drop(columns=["discourse_type"])
    test_df = test_df.rename(columns={"discourse_effectiveness": "label"})

    # reorganize by essay_id
    essay_ids = train_df.essay_id.unique()
    np.random.seed(42)
    np.random.shuffle(essay_ids)
    val_prop = 1 - split_ratio
    val_sz = int(len(essay_ids) * val_prop)
    val_essay_ids = essay_ids[:val_sz]

    is_val = np.isin(train_df.essay_id, val_essay_ids)
    idxs = np.arange(len(train_df))
    val_idxs = idxs[is_val]
    trn_idxs = idxs[~is_val]
    train = train_df.iloc[trn_idxs]
    valid = train_df.iloc[val_idxs]

    # create new train, valid, and test data set
    train.to_csv(os.path.join(current_path + data_dir,
                              "train_new.csv"), index=False)
    valid.to_csv(os.path.join(current_path + data_dir,
                              "valid_new.csv"), index=False)
    test_df.to_csv(os.path.join(current_path + data_dir,
                                "test_new.csv"), index=False)

    TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    train_fields = [("discourse_id", None), ("essay_id", None), ('discourse_text', TEXT),
                    ('label', LABEL)]
    valid_fields = [("discourse_id", None), ("essay_id", None), ('discourse_text', TEXT),
                    ('label', LABEL)]
    test_fields = [("discourse_id", None), ("essay_id", None),
                   ('discourse_text', TEXT)]

    train_data = data.TabularDataset(path=os.path.join(
        current_path + data_dir, "train_new.csv"), format='csv', fields=train_fields, skip_header=True)
    valid_data = data.TabularDataset(path=os.path.join(
        current_path + data_dir, "valid_new.csv"), format='csv', fields=valid_fields, skip_header=True)
    test_data = data.TabularDataset(path=os.path.join(
        current_path + data_dir, "test_new.csv"), format='csv', fields=test_fields, skip_header=True)

    # In order to use BERT with torchtext, we have to set use_vocab=Fasle such that the torchtext knows we will not be building
    # our own vocabulary using our dataset from scratch. Instead, use pre-trained BERT tokenizer and its corresponding
    # word-to-index mapping.

    TEXT.build_vocab(train_data, min_freq=3, vectors=vectors)
    LABEL.build_vocab(train_data)

    if data_information:
        print("Training Data information")
        print(type(TEXT.vocab))
        # No. of unique tokens in text
        print("Size of TEXT vocabulary:", len(TEXT.vocab))

        # No. of unique tokens in label
        print("Size of LABEL vocabulary:", len(LABEL.vocab))

        # Commonly used words
        print("Ten most commly used words are",
              TEXT.vocab.freqs.most_common(10))

        # Word dictionary
        # print(TEXT.vocab.stoi)
        # print(LABEL.vocab.stoi)
    # check whether cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load an iterator
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.discourse_text),
        sort_within_batch=True,
        shuffle=True,
        device=device)
    test_iterator = data.BucketIterator(
        test_data,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.discourse_text),
        sort_within_batch=True,
        device=device)

    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL
