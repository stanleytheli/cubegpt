import torch
import numpy as np
from utils import *

def scrambled(n):
    """Returns a CubeState scrambled over n random moves."""
    state = CubeState()
    for i in range(n):
        state = state.get_random_child()
    return state

def generate_batch(device, n, scramble_limit = 21):
    x = []
    y = []
    for i in range(n):
        steps = np.random.randint(scramble_limit)
        x.append(scrambled(steps).get_tokens())
        y.append(steps)
    x = torch.stack(x).to(device)
    y = torch.tensor(y).to(device)
    return x, y

def generate_path_batch(device, k, k_max = 21):
    x = []
    y = []
    for i in range(k):
        curr = CubeState()
        for steps in range(k_max):
            x.append(curr.get_tokens())
            y.append(steps)
            curr = curr.get_random_child()
    x = torch.stack(x).to(device)
    y = torch.tensor(y).to(device)
    return x, y

