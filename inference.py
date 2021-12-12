import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from time import perf_counter
import numpy as np


class Net(nn.Module):
    def __init__(self, input_feature, num_hidden, outputs):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_feature, num_hidden)
        self.out = nn.Linear(num_hidden, outputs)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.out(x)
        return x


inference = Net(input_feature=6, num_hidden=40, outputs=1)
inference.load_state_dict(torch.load("model2.pkl"))

import pickle

with open('data/dataset/dataset_tmp.pkl', 'rb') as f:
    dataset = pickle.load(f)
dataset_tensor = torch.tensor(dataset)
x = dataset_tensor[:, :6].to(torch.float32)
y = torch.unsqueeze(dataset_tensor[:, 6], dim=1).to(torch.float32)
print(inference(x[100:200,:]))
print(y[100:2000])

