import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.7)

        self.hidden_layer1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.7)

        # self.hidden_layer2 = nn.Linear(hidden_size2, hidden_size3)
        # self.relu3 = nn.ReLU()

        self.output_layer = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.hidden_layer1(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        # out = self.hidden_layer2(out)
        # out = self.relu3(out)
        out = self.output_layer(out)
        out = self.softmax(out)
        return out
