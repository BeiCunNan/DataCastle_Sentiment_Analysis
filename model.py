import torch
from torch import nn


# Bert + FNN
class Transformer(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        # The pooler_output is made of CLS --> FNN --> Tanh
        # The last_hidden_state[:,0] is made of original CLS
        # Method one
        # cls_feats  = raw_outputs.pooler_output
        # Method two
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.softmax(self.linear(self.dropout(cls_feats)))
        return predicts

class Transformer_Extend_LSTM(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_outputs, _ = self.lstm(tokens)

        lstm_outputs = lstm_outputs[:, -1, :]
        predicts = self.block(lstm_outputs)
        return predicts



class Gru_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.Gru = nn.GRU(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        gru_output, _ = self.Gru(tokens)
        outputs = gru_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


# Try to use the softmax、relu、tanh and logistic
class Lstm_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.Lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=320,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_output, _ = self.Lstm(tokens)
        outputs = lstm_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class BiLstm_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        # Open the bidirectional
        self.BiLstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=320,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320 * 2, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.BiLstm(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class Rnn_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.Rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.Rnn(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs
