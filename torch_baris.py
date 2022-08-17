import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np


# Inport data and transform to torch

df = pd.read_csv('data.csv', header=0)
data_raw = df.to_numpy()

survived = data_raw[:,23]
input_features = data_raw[:,0:23]

## Split the data 

X_train, X_test, y_train, y_test = train_test_split(input_features, survived, test_size=0.1, random_state=39)

#X_train = X_train.astype(np.float32)
X_train_trc = torch.from_numpy(X_train.astype(np.float32))
X_test_trc = torch.from_numpy(X_test)

y_train_trc = torch.from_numpy(y_train.astype(np.float32))
y_test_trc = torch.from_numpy(y_train)
# NN part

class NeuralNetwork(nn.Module):
    def _init_(self):
        super(NeuralNetwork, self)._init_()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(23, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

#X = torch.rand(1, 28, 28, device=device)
#X = X_train_trc[0:1]

X = X_train_trc

logits = model(X)

print(logits)

pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
y_pred = pred_probab.argmax(1)


#
print(f"Predicted class: {y_pred}")

# loss function - CrossEntropyLoss expects floating point inputs and long labels.

y_train_trc_new = y_train_trc.reshape((y_train_trc.shape[0], 1))
print(y_train_trc_new)
loss_fn = torch.nn.CrossEntropyLoss()

y_pred_new=y_pred.reshape((y_pred.shape[0], 1))

loss = loss_fn(y_pred_new.float(), y_train_trc_new.float())

print('Total loss for this batch: {}'.format(loss.item()))