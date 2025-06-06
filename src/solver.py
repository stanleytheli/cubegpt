import torch
from torch import nn
from utils import *
from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerClassifier(activation=nn.functional.gelu)
model.load_state_dict(torch.load("C:/Users/stanl/cubegpt/models/cubegpt-cls-t51m.pth", 
                                 map_location=torch.device("cuda")))
model = model.to(device)
model = model.eval()

def beam_step(frontier : list[CubeState], width = 100):
    new_frontier = []

    for frontier_state in frontier:
        for child in frontier_state.get_children():
            new_frontier.append(child)

    if len(new_frontier) > width:
        batch_size = 256
        n_batches = int(np.ceil(len(new_frontier) / batch_size))
        state_evals = {}
        for b in range(n_batches):
            batch = torch.stack([c.get_tokens().to(device) 
                                    for c in new_frontier[b*batch_size:(b+1)*batch_size]])
            with torch.no_grad():
                evaluations = model.estimate(batch).detach()

            # log results for this batch
            for frontier_state, model_eval in zip(new_frontier[b*batch_size:(b+1)*batch_size], 
                                                    evaluations):
                state_evals[frontier_state.cube_string] = model_eval
                
        new_frontier = sorted(new_frontier, key = lambda state : state_evals[state.cube_string])[:width]
        new_frontier = new_frontier[:width]
    
    return new_frontier

def beam_search(state : CubeState, 
                width = 4000, 
                max_moves=30, 
                verbose=False):
    with torch.no_grad():        
        state.forget()
        frontier : list[CubeState] = [state]
        history : list[list[CubeState]] = [frontier]

        for _ in range(max_moves):
            frontier = beam_step(frontier, width)
            history.append(frontier)

            # monitoring
            if verbose:
                print(frontier[0].cube_string)

            for frontier_state in frontier:
                if frontier_state.cube_string == solved_string:
                    return True, frontier_state, history
                    
        return False, frontier[0], history
    

def beam_search_with_tightening(state : CubeState, 
                                start_width = 200, 
                                tightening_factor = 0.9,
                                min_width = 100,
                                max_moves=30, 
                                verbose=False):
    """Just like beam search, but with exponentially decaying width."""
    with torch.no_grad():        
        state.forget()
        frontier : list[CubeState] = [state]
        history : list[list[CubeState]] = [frontier]
        current_width = start_width

        for _ in range(max_moves):
            frontier = beam_step(frontier, current_width)
            history.append(frontier)

            # monitoring
            if verbose:
                print(frontier[0].cube_string)

            for frontier_state in frontier:
                if frontier_state.cube_string == solved_string:
                    return True, frontier_state
            
            current_width = max(min_width, int(current_width * tightening_factor))
        return False, frontier[0]
    

def display(state):
    cube_strings = []
    steps = []
    curr = state
    while curr is not None:
        steps = [curr.prev_move] + steps
        cube_strings = [curr.cube_string] + cube_strings
        curr = curr.prev_state
    for step in steps[1:]:
        print(step, " ",end="")
    print("")
    print(len(steps) - 1)
    for i in range(len(steps)):
        if i != 0:
            print(steps[i])
        print(cube_strings[i])

input_cubestring = "YGGRWYRRWGBWYOOWGRGWBWGBBOOOOOWRBYGRYORYBRWYOYBBWYRGGB"
state = CubeState(input_cubestring)
solved, final_state, history = beam_search(state)
display(final_state)