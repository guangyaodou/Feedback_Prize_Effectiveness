import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import models as U
import bert_preprocess as bp
import preprocessing as preprocess
import train as train
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# DO NOT Modify
EMBEDDING_DIM = 100
NUM_CLASSES = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# settings
settings_path = 'config.yml'
settings = yaml.safe_load(open(settings_path, "r"))

# ("RNN", "LSTM", "GRU", "BERT", "DEBERT")
model_name = settings["general"]["model"].upper()

if model_name == "BERT" or model_name == "DEBERT":
    print(f"Running model {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        settings[model_name]["model_nm"], num_labels=NUM_CLASSES)
    train_dataset, valid_dataset, _ = bp.load_data(model_nm=settings[model_name]["model_nm"],
                                                   split_ratio=settings[model_name]["split_ratio"],
                                                   data_information=settings["general"]["data_information"])
    tokenizer = AutoTokenizer.from_pretrained(settings[model_name]["model_nm"])
    trainer = train.bert_trainer(model=model,
                                 train_data=train_dataset,
                                 valid_data=valid_dataset,
                                 tokenizer=tokenizer,
                                 epochs=settings[model_name]["EPOCH_NUM"],
                                 wd=settings[model_name]["weight_decay"],
                                 bs=settings[model_name]["BATCH_SIZE"],
                                 lr=float(settings[model_name]
                                          ["learning_rate"]),
                                 model_name=model_name)
    trainer.train()
else:
    train_iterator, valid_iterator, test_iterator, TEXT, LABEL = preprocess.load_data(
        BATCH_SIZE=settings[model_name]["BATCH_SIZE"],
        split_ratio=settings[model_name]["split_ratio"],
        data_information=settings["general"]["data_information"])

    # define hyperparameters
    embedding_dim = EMBEDDING_DIM
    num_classes = NUM_CLASSES

    # instantiate the model
    if model_name == "RNN":
        print("Running model", model_name)
        model = U.RNN(TEXT=TEXT,
                      embedding_size=embedding_dim,
                      hidden_size=settings[model_name]["hidden_size"],
                      num_classes=num_classes,
                      layers=settings[model_name]["number_of_layers"],
                      dropout=settings[model_name]["dropout"],
                      device=device,
                      bidirectional=settings[model_name]["bidirectional"],
                      last_hidden=settings[model_name]["last_hidden"])
    elif model_name == "LSTM":
        print("Running model", model_name)
        model = U.LSTM(TEXT=TEXT,
                       embedding_size=embedding_dim,
                       hidden_size=settings[model_name]["hidden_size"],
                       num_classes=num_classes,
                       layers=settings[model_name]["number_of_layers"],
                       dropout=settings[model_name]["dropout"],
                       device=device,
                       bidirectional=settings[model_name]["bidirectional"],
                       last_hidden=settings[model_name]["last_hidden"])
    elif model_name == "GRU":
        print("Running model", model_name)
        model = U.GRU(TEXT=TEXT,
                      embedding_size=embedding_dim,
                      hidden_size=settings[model_name]["hidden_size"],
                      num_classes=num_classes,
                      layers=settings[model_name]["number_of_layers"],
                      dropout=settings[model_name]["dropout"],
                      device=device,
                      bidirectional=settings[model_name]["bidirectional"],
                      last_hidden=settings[model_name]["last_hidden"])
    else:
        raise Exception("INVALID MODEL")

    # Set criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(settings[model_name]["learning_rate"]),
                           weight_decay=float(settings[model_name]["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10)
    
    # Train the model and get accuracy info
    train_loss, train_accuracy, valid_loss, valid_accuracy, net = train.model_train(net=model,
                                                                                    train_iterator=train_iterator,
                                                                                    valid_iterator=valid_iterator,
                                                                                    epoch_num=settings[model_name][
                                                                                        "EPOCH_NUM"],
                                                                                    criterion=criterion,
                                                                                    optimizer=optimizer,
                                                                                    scheduler=None,
                                                                                    device=device)

    train.plot_loss(train_loss=train_loss,
                    valid_loss=valid_loss, model_name=model_name)
    train.plot_accuracy(train_accuracy=train_accuracy,
                        valid_accuracy=valid_accuracy, model_name=model_name)
