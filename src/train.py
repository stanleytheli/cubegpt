import numpy as np
import pandas as pd
import torch
from torch import nn
from utils import *

from model import CubeTransformer
from data_generation import *

import time
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_batch(X : torch.Tensor, y : torch.Tensor, model : nn.Module, loss_fn : nn.Module):
    pred = model(X)
    loss = loss_fn(pred, y.float())
    total_loss = loss.item() * len(y)
    return total_loss

def train_batch(X : torch.Tensor, 
                y : torch.Tensor, 
                model : nn.Module, 
                loss_fn : nn.Module, 
                optimizer : torch.optim.Optimizer):
    # Forward pass
    pred = model(X)
    loss = loss_fn(pred, y.float())
    total_loss = loss.item() * len(y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return total loss
    return total_loss


model = CubeTransformer()
# model = nn.DataParallel(model) # If there are multiple GPUs
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3000)
model = model.to(device)

epochs = 30000
batch_size = 128
scramble_limit = 21
test_sample_batches = 10
test_interval = 100
last_time = time.time()

losses = []

for epoch in range(epochs):
    X, y = generate_batch(device, batch_size, scramble_limit)
    avg_train_loss = train_batch(X, y, model, loss_fn, optim) / len(y)
    scheduler.step(avg_train_loss)

    if (epoch + 1) % test_interval == 0:
        test_loss = 0
        for sample in range(test_sample_batches):
            Xt, yt = generate_batch(device, batch_size, scramble_limit)
            test_loss += test_batch(Xt, yt, model, loss_fn)
        avg_loss = test_loss / (test_sample_batches * batch_size)
        losses.append(avg_loss)
    
        now = time.time()
        print(f"Epoch {epoch + 1}: avg loss {avg_loss} / took {now - last_time} s / current LR {scheduler.get_last_lr()}")
        last_time = now

torch.save(model, "./models/new_model.pth")