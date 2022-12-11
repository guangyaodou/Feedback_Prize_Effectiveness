import matplotlib.pyplot as plt
import spacy
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from transformers import TrainingArguments, Trainer


def model_train(net, train_iterator, valid_iterator, epoch_num, criterion, optimizer, scheduler, device):
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    net = net.to(device)
    for epoch in range(epoch_num):
        print("Running Epoch ", str(epoch + 1))
        net.train()
        train_acc = 0
        train_l = 0
        valid_acc = 0
        valid_l = 0
        score = 0
        for i, batch in enumerate(train_iterator):
            text, text_len = batch.discourse_text
            text = text.to(device)
            text_len = text_len.to(device)
            y = batch.label.to(device)
            optimizer.zero_grad()
            y_hat = net(text, text_len)
            l = criterion(y_hat, y.to(dtype=torch.long))
            train_acc += calculate_accuracy(y_hat, y, device)
            l.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + i / len(train_iterator))
            train_l += l
        train_acc = train_acc / len(train_iterator)
        train_acc = train_acc.cpu().detach().numpy()
        train_accuracy.append(train_acc)
        train_l = train_l / len(train_iterator)

        train_l = train_l.cpu().detach().numpy()
        train_loss.append(train_l)
        if epoch == 0 or (epoch + 1) % 2 == 0:
            print(f"train loss after epoch {epoch + 1} is {train_l}")
            print(f"train accuracy after epoch {epoch + 1} is {train_acc}")

        net.eval()
        with torch.no_grad():
            for batch in valid_iterator:
                text, text_len = batch.discourse_text
                text = text.to(device)
                text_len = text_len.to(device)
                y = batch.label.to(device)

                y_hat = net(text, text_len)
                l = criterion(y_hat, y.to(dtype=torch.long))
                valid_acc += calculate_accuracy(y_hat, y, device)
                valid_l += l
                score += get_score(y_hat, y)

            score /= len(valid_iterator)
            valid_acc = valid_acc / len(valid_iterator)

            valid_acc = valid_acc.cpu().detach().numpy()
            valid_accuracy.append(valid_acc)

            valid_l = valid_l / len(valid_iterator)

            valid_l = valid_l.cpu().detach().numpy()
            valid_loss.append(valid_l)

            if epoch == 0 or (epoch + 1) % 2 == 0:
                print(f"valid loss after epoch {epoch + 1} is {valid_l}")
                print(f"valid accuracy after epoch {epoch + 1} is {valid_acc}")
                print(f"the score after epoch {epoch + 1} is  {score}")
    return train_loss, train_accuracy, valid_loss, valid_accuracy, net


def bert_trainer(model, train_data, valid_data, tokenizer, epochs, wd, bs, lr, model_name):
    args = TrainingArguments(f'../output/{model_name}', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
                             evaluation_strategy="epoch", per_device_train_batch_size=bs,
                             per_device_eval_batch_size=bs * 2,
                             num_train_epochs=epochs, weight_decay=wd, report_to='none')
    return Trainer(model, args, train_dataset=train_data, eval_dataset=valid_data,
                   tokenizer=tokenizer, compute_metrics=score)


def score(preds):
    return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}


def calculate_accuracy(y_hat, y, device):
    y_hat = y_hat.to(device)
    y = y.to(device)
    correct = 0
    y_hat = F.softmax(y_hat, dim=1).argmax(1)
    correct += (y_hat == y.squeeze()).sum()
    return correct / y_hat.shape[0]


def get_score(outputs, labels):
    outputs = F.softmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu()
    labels = F.one_hot(labels.to(torch.int64), num_classes=3)
    score = log_loss(labels, outputs)
    return round(score, 5)


def inference(net, sentence, TEXT, LABEL, device):
    nlp = spacy.load('en_core_web_sm')
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  # tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    text = torch.LongTensor(indexed).to(device)  # convert to tensor
    text = text.unsqueeze(1).T  # reshape in form of batch * no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = net(text, length_tensor)  # prediction
    prediction = F.softmax(prediction, dim=1).argmax(1)
    for key in LABEL.vocab.stoi:
        if LABEL.vocab.stoi[key] == prediction.item():
            result = key
    return result


def plot_loss(train_loss, valid_loss, model_name):
    f, ax = plt.subplots(1)
    ax.plot(train_loss, label="Training Loss")
    ax.plot(valid_loss, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Loss vs Epoch')
    legend = plt.legend(loc='lower right')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.savefig(f'../output/{model_name}_Training_Validation_Loss.png', bbox_inches='tight', dpi=2000)
    plt.show()


def plot_accuracy(train_accuracy, valid_accuracy, model_name):
    f, ax = plt.subplots(1)
    ax.plot(train_accuracy, label="Training Accuracy")
    ax.plot(valid_accuracy, label=" Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('Accuracy vs Epoch')
    legend = plt.legend(loc='lower right')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.savefig(f'../output/{model_name}_Training_Validation_Accuracy.png', bbox_inches='tight', dpi=2000)
    plt.show()
