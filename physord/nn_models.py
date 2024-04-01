# This is modified from https://github.com/Physics-aware-AI/Symplectic-ODENet/blob/master/nn_models.py

import torch
import numpy as np
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class ForceMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForceMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class MLP(torch.nn.Module):
    '''Multilayer perceptron'''
    def __init__(self, input_dim, hidden_dim, output_dim, bias_bool=True):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.kaiming_normal_(l.weight, mode='fan_in', nonlinearity='tanh')
            if l.bias is not None:
                torch.nn.init.constant_(l.bias, 0)

        self.nonlinearity = torch.tanh

    def forward(self, x):
        h = self.nonlinearity( self.linear1(x) )
        h = self.nonlinearity( self.linear2(h) )
        return self.linear3(h)

def FixedMass(m_dim, eps, param_value):

    off_diag_dim = int(m_dim * (m_dim - 1) / 2)

    diag = param_value * torch.ones(m_dim)
    off_diag = torch.zeros(off_diag_dim)

    L = torch.diag_embed(diag)

    ind = np.tril_indices(m_dim, k=-1)
    flat_ind = np.ravel_multi_index(ind, (m_dim, m_dim))
    L = torch.flatten(L, start_dim=0)
    L[flat_ind] = off_diag
    L = torch.reshape(L, (1, m_dim, m_dim))

    M = torch.bmm(L, L.permute(0, 2, 1))
    for i in range(m_dim):
        M[:, i, i] = M[:, i, i] + eps[i]

    return M

def FixedInertia(m_dim, eps, param_value):
    off_diag_dim = int(m_dim * (m_dim - 1) / 2)
    # eps = torch.tensor(eps)

    params = param_value * torch.ones(off_diag_dim + m_dim)

    diag, off_diag = torch.split(params, [m_dim, off_diag_dim], dim=0)
    L = torch.diag_embed(diag)

    ind = np.tril_indices(m_dim, k=-1)
    flat_ind = np.ravel_multi_index(ind, (m_dim, m_dim))
    L = torch.flatten(L, start_dim=0)
    L[flat_ind] = off_diag
    L = torch.reshape(L, (1, m_dim, m_dim))

    M = torch.bmm(L, L.permute(0, 2, 1))
    for i in range(m_dim):
        M[:, i, i] += eps[i]
    return M

