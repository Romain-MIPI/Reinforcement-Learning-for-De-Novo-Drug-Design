import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MLP(nn.Module):
    """Base class for MLP module"""
    def __init__(self, params):
        super(MLP, self).__init__()
        self.params = params
        self.hidden_size = self.params['hidden_size']
        self.input_size = [self.params['input_size']] + self.hidden_size[:-1]
        self.n_layers = self.params['n_layers']
        self.activation = self.params['activation']
        if type(self.activation) is list:
            assert len(self.activation) == self.n_layers
        else:
            self.activation = [self.activation] * self.n_layers
        if 'dropout' in self.params.keys():
            self.dropout = self.params['dropout']
        else:
            self.dropout = 0
        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        for i in range(self.n_layers - 1):
            self.dropouts.append(nn.Dropout(self.dropout))
            self.bn.append(nn.BatchNorm1d(self.hidden_size[i]))
            self.layers.append(nn.Linear(in_features=self.input_size[i], out_features=self.hidden_size[i]))
        i = self.n_layers - 1
        self.dropouts.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(in_features=self.input_size[i], out_features=self.hidden_size[i]))

    def forward(self, inp):
        output = inp
        for i in range(self.n_layers - 1):
            output = self.dropouts[i](output)
            output = self.layers[i](output)
            output = self.bn[i](output)
            output = self.activation[i](output)
        output = self.dropouts[-1](output)
        output = self.layers[-1](output)
        output = self.activation[-1](output)
        return output