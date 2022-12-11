import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, TEXT, embedding_size, hidden_size, num_classes, layers,
                 dropout, device, bidirectional=True, last_hidden=True):
        super(RNN, self).__init__()
        self.last_hidden = last_hidden
        self.device = device
        self.bidirectional = bidirectional
        self.vocab_size = len(TEXT.vocab)

        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=layers,
                          dropout=dropout,
                          bidirectional=bidirectional)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, 64)
        else:
            self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, text, text_len):
        # dim(text) = batch_size * sentence_length
        x = self.embedding(text).to(self.device)
        # dim(x) = batch_size * L * embedding_size

        # packed sequence
        packed_sequence = nn.utils.rnn.pack_padded_sequence(x, text_len.cpu(), batch_first=True)

        output, hidden = self.rnn(packed_sequence)
        # dim(hn) = (D*num_layers) * batch_size * H out
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # dim(padded_output) = L * batch_size * (D*H out)

        if self.last_hidden:
            if self.bidirectional:
                y = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                y = hidden[-1]
            # dim(y) = batch_size * D*(H out)
        else:
            # dim(padded_output) = L * batch_size * (D*H out)
            # Avg pooling on padded_output's L's dimension
            y = torch.mean(padded_output, dim=0)
            # dim(y) = batch_size * D*(H out)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=0.2)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=0.2)
        y = torch.sigmoid(self.fc3(y))
        return y


class LSTM(nn.Module):
    def __init__(self, TEXT, embedding_size, hidden_size, num_classes, layers,
                 dropout, device, bidirectional=True, last_hidden=True):
        super(LSTM, self).__init__()
        self.last_hidden = last_hidden
        self.device = device
        self.bidirectional = bidirectional
        self.vocab_size = len(TEXT.vocab)

        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, 64)
        else:
            self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, text, text_len):
        # dim(text) = batch_size * sentence_length
        x = self.embedding(text).to(self.device)
        # dim(x) = batch_size * L * embedding_size

        # packed sequence
        packed_sequence = nn.utils.rnn.pack_padded_sequence(x, text_len.cpu(), batch_first=True)

        output, (hidden, _) = self.lstm(packed_sequence)
        # dim(hn) = (D*num_layers) * batch_size * H out
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # dim(padded_output) = L * batch_size * (D*H out)

        if self.last_hidden:
            if self.bidirectional:
                y = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                y = hidden[-1]
            # dim(y) = batch_size * D*(H out)
        else:
            # dim(padded_output) = L * batch_size * (D*H out)
            # Avg pooling on padded_output's L's dimension
            y = torch.mean(padded_output, dim=0)
            # dim(y) = batch_size * D*(H out)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=0.2)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=0.2)
        y = torch.sigmoid(self.fc3(y))
        return y


class GRU(nn.Module):
    def __init__(self, TEXT, embedding_size, hidden_size, num_classes, layers,
                 dropout, device, bidirectional=True, last_hidden=True):
        super(GRU, self).__init__()
        self.last_hidden = last_hidden
        self.device = device
        self.bidirectional = bidirectional
        self.vocab_size = len(TEXT.vocab)

        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
        self.gru = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=layers,
                          dropout=dropout,
                          bidirectional=bidirectional)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, 64)
        else:
            self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, text, text_len):
        # dim(text) = batch_size * sentence_length
        x = self.embedding(text).to(self.device)
        # dim(x) = batch_size * L * embedding_size

        # packed sequence
        packed_sequence = nn.utils.rnn.pack_padded_sequence(x, text_len.cpu(), batch_first=True)

        output, hidden = self.gru(packed_sequence)
        # dim(hn) = (D*num_layers) * batch_size * H out
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # dim(padded_output) = L * batch_size * (D*H out)

        if self.last_hidden:
            if self.bidirectional:
                y = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                y = hidden[-1]
            # dim(y) = batch_size * D*(H out)
        else:
            # dim(padded_output) = L * batch_size * (D*H out)
            # Avg pooling on padded_output's L's dimension
            y = torch.mean(padded_output, dim=0)
            # dim(y) = batch_size * D*(H out)
        y = F.relu(self.bn1(self.fc1(y)))
        y = F.dropout(y, p=0.2)
        y = F.relu(self.bn2(self.fc2(y)))
        y = F.dropout(y, p=0.2)
        y = torch.sigmoid(self.fc3(y))
        return y
