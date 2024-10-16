import torch
from torch import nn as nn

# Modules
class RNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1,bidirectional=False):
        super(RNNModule, self).__init__()

        # 기본 RNN 레이어 추가
        self.rnn = nn.RNN(input_dim, hidden_dim, bidirectional=bidirectional, dropout=rec_dropout, batch_first=True, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent

class LSTMModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True,
                           dropout=rec_dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent

class GRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRUModule, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True,
                          dropout=rec_dropout, num_layers=num_layers)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent

# Models
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(RNN, self).__init__()
        self.rnn_module = RNNModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.rnn_module(x)
        out = self.fc(recurrent)
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm_module = LSTMModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.lstm_module(x)
        out = self.fc(recurrent)
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, rec_dropout=0.4, num_layers=1, bidirectional=False):
        super(GRU, self).__init__()
        self.gru_module = GRUModule(input_dim, hidden_dim, rec_dropout, num_layers, bidirectional)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        recurrent = self.gru_module(x)
        out = self.fc(recurrent)
        return out