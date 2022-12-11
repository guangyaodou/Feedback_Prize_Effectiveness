import logging
import warnings

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def load_data(model_nm, split_ratio, data_information=False):
    df = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    warnings.simplefilter('ignore')
    logging.disable(logging.WARNING)

    tokz = AutoTokenizer.from_pretrained(model_nm)
    sep = tokz.sep_token
    df['inputs'] = df.discourse_type + sep + df.discourse_text
    df_test['inputs'] = df_test.discourse_type + sep + df_test.discourse_text

    new_label = {"discourse_effectiveness": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
    df = df.replace(new_label)
    df = df.rename(columns={"discourse_effectiveness": "label"})

    ds = Dataset.from_pandas(df)
    ds_test = Dataset.from_pandas(df_test)

    def tok_func(x):
        return tokz(x["inputs"], truncation=True)

    inps = "discourse_text", "discourse_type"
    # tok_ds = ds.map(tok_func, batched=True) Dataset({
    #     features: ['discourse_id', 'essay_id', 'discourse_text', 'discourse_type', 'label', 'inputs', 'input_ids', 'token_type_ids', 'attention_mask'],
    #     num_rows: 36765
    # })
    # tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','discourse_id','essay_id'))
    tok_ds = ds.map(tok_func, batched=True, remove_columns=inps + ('inputs', 'discourse_id', 'essay_id'))
    #  Dataset({
    #     features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
    #     num_rows: 36765
    # })

    if data_information:
        print(ds.info)

    essay_ids = df.essay_id.unique()
    np.random.seed(42)
    np.random.shuffle(essay_ids)

    val_prop = 1 - split_ratio
    val_sz = int(len(essay_ids) * val_prop)
    val_essay_ids = essay_ids[:val_sz]  # the essay ids of validation sets

    is_val = np.isin(df.essay_id, val_essay_ids)  # checks whether the essay id belongs to the validation sets
    idxs = np.arange(len(df))
    val_idxs = idxs[is_val]
    trn_idxs = idxs[~is_val]

    dds = DatasetDict({"train": tok_ds.select(trn_idxs),
                       "valid": tok_ds.select(val_idxs)})

    train = dds["train"]
    valid = dds["valid"]

    return train, valid, ds_test
