import torch
from torch import nn
from utils import *
from model import CubeTransformer

import numpy as np
from train import *
from solver import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerClassifier(activation=nn.functional.gelu)
model.load_state_dict(torch.load("C:/Users/stanl/cubegpt/models/cubegpt-cls-t51m.pth", 
                                 map_location=torch.device("cuda")))
model = model.to(device)
model = model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_loss(num_batches=250, batch_size=128):
    test_losses = []
    with torch.no_grad():
        for sample in range(num_batches):
            Xt, yt = generate_batch(batch_size, scramble_limit)
            test_losses.append(test_batch(Xt, yt, model, loss_fn))
    avg_loss = np.mean(test_losses) / (batch_size)
    std_loss = np.std(test_losses) / (batch_size * np.sqrt(num_batches))
    return avg_loss, std_loss

def measure_acc(num_batches, batch_size, tolerance):
    list_mode = False
    if type(tolerance) == list:
        list_mode = True
    
    if list_mode:
        total_correct = [0] * len(tolerance)
    else:
        total_correct = 0
    
    sample_size = num_batches * batch_size

    with torch.no_grad():
        for i in range(num_batches):
            X, y = generate_batch(batch_size, scramble_limit=21)
            output = model.estimate(X).detach()
            error = torch.abs(output - y)

            if list_mode:
                for index, target in enumerate(tolerance):
                    total_correct[i] += torch.sum(error < target).item()
            else:
                total_correct += torch.sum(error < tolerance).item()

    if list_mode:
        p = [0] * len(total_correct)
        std_p = [0] * len(total_correct)
        for i in total_correct:
            p[i] = total_correct[i] / sample_size
            std_p[i] = np.sqrt(p * (1 - p) / sample_size)
    else:
        p = total_correct / sample_size
        std_p = np.sqrt(p * (1 - p) / sample_size)

    return p, std_p

def measure_solving(trials, scramble_depth, width, max_moves, verbose=False):
    total_successes = 0
    solution_lengths = []
    for t in range(trials):
        random = scrambled(scramble_depth)
        solved, final_state, history = beam_search(random, width, max_moves=max_moves)
        total_successes += solved
        if solved:
            solution_lengths.append(len(history) - 1)
        if verbose:
            if (t+1) % 5 == 0:
                print(f"trial {t+1} done")

    p_solve = total_successes / trials
    std_p_solve = np.sqrt(p_solve * (1 - p_solve) / trials)
    avg_len = np.mean(solution_lengths)
    std_len = np.std(solution_lengths)

    return p_solve, std_p_solve, avg_len, std_len