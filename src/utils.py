import torch
import json
import numpy as np

solved_string = "WWWWWWWWWOOOOOOOOOGGGGGGGGGRRRRRRRRRBBBBBBBBBYYYYYYYYY"

f = open("C:/Users/stanl/cubegpt/lookup_tables/cube_utils.json", "r")
tables = json.load(f)
f.close()

pos_idx = eval(tables["pos_idx"])
pos_face_idx = eval(tables["pos_face_idx"])
color_str_tensor_idx = tables["color_str_tensor_idx"]
color_str_orientation = tables["color_str_orientation"]
move_map = tables["move_map"]

def get_colors_str_by_pos(cube_string, position):
    return "".join([cube_string[i] for i in pos_face_idx[position]])

def get_inverse_move(move):
    """``move`` must be in the set ``{"U", "U'", "U2", "D"...}``"""
    if len(move) == 1:
        return move + "'"
    if move[1] == "'":
        return move[0]
    if move[1] == "2":
        return move

def string_to_tokens(cube_string):
    tokens_tensor = torch.zeros((20, 2), dtype=torch.int32)

    orien_idx_to_token = {
        0: 0,
        3: 1,
        4: 2,
        1: 0,
        2: 1,
        5: 2,
    }

    edge_orientations = 2
    corner_orientations = 3

    for pos, position_idx in pos_idx.items():
        color_str = get_colors_str_by_pos(cube_string, pos)
        is_edge = len(color_str) == 2
        
        piece_idx = color_str_tensor_idx[color_str]
        orientation_idx = color_str_orientation[color_str]

        if is_edge:
            placement_token = position_idx * edge_orientations + orientation_idx
        else:
            orientation_idx = orien_idx_to_token[orientation_idx]
            placement_token = position_idx * corner_orientations + orientation_idx
            placement_token += 24

        tokens_tensor[piece_idx][0] = piece_idx # redundant, but makes code more consistent.
        tokens_tensor[piece_idx][1] = placement_token   

    return tokens_tensor

def string_to_tensor(cube_string):
    cube_tensor = torch.zeros((20, 14), dtype=torch.float32)
    
    for pos, pos_value in pos_idx.items():
        color_str = get_colors_str_by_pos(cube_string, pos) 
        is_edge = len(color_str) == 2

        tensor_idx = color_str_tensor_idx[color_str]
        orientation_value = color_str_orientation[color_str] 

        cube_tensor[tensor_idx][pos_value] = 1
        if is_edge:
            cube_tensor[tensor_idx][12 + orientation_value] = 1
        else:
            cube_tensor[tensor_idx][8 + orientation_value] = 1
    
    return cube_tensor


def rotate_string(cube_str, move=None):
    """Fast rotate function for cube strings."""    
    return "".join([cube_str[i] for i in move_map[move]])
    
class CubeState:
    # moves[move_class][move_type]
    # move_class --> U, U', U2, D, D', D2 
    # move_type --> U, U', U2
    # when scrambling:
    # maximum 2 moves with same move_class in a row
    # maximum 1 move with same move_class, move_type in a row 
    moves = [
        [["U", "U'", "U2"], ["D", "D'", "D2"]],
        [["L", "L'", "L2"], ["R", "R'", "R2"]],
        [["F", "F'", "F2"], ["B", "B'", "B2"]],
             ]

    def __init__(self, cube_string : str = solved_string,
                 prev_state = None,
                 prev_move = None,
                 prev_move_class = None,
                 prev_move_type = None
                 ):
        self.cube_string = cube_string
        self.prev_state = prev_state
        self.prev_move = prev_move
        self.prev_move_class = prev_move_class
        self.prev_move_type = prev_move_type

        self.prev2_move_class = None
        if prev_state:
            self.prev2_move_class = prev_state.prev_move_class
    
    def forget(self):
        """Forget this Cube's history (for fair solving)."""
        self.prev_state = None
        self.prev_move = None
        self.prev_move_class = None
        self.prev_move_type = None
        self.prev2_move_class = None

    def get_tensor(self):
        return string_to_tensor(self.cube_string)
    
    def get_tokens(self):
        return string_to_tokens(self.cube_string)

    def get_allowed_moves(self):
        # (move_class, move_type, move_index)
        for move_class in range(len(CubeState.moves)):
            # Skip if this would be the 3rd consecutive move from the same move class
            if (move_class == self.prev_move_class and 
                move_class == self.prev2_move_class):
                continue

            for move_type in range(len(CubeState.moves[move_class])):
                # Skip if this would be the 2nd consecutive move from the same move type
                if (move_class == self.prev_move_class and 
                    move_type == self.prev_move_type):
                    continue

                for move_index in range(len(CubeState.moves[move_class][move_type])):
                    yield (move_class, move_type, move_index)

    def get_child(self, move_class, move_type, move_index):
        """Get child state with given move_class, move_type, and move_index"""
        move = CubeState.moves[move_class][move_type][move_index]
        return CubeState(
            cube_string=rotate_string(self.cube_string, move),
            prev_state=self,
            prev_move=move,
            prev_move_class=move_class,
            prev_move_type=move_type
        )

    def get_random_child(self):
        allowed_moves = list(self.get_allowed_moves())
        move_class, move_type, move_index = allowed_moves[np.random.randint(len(allowed_moves))]
        return self.get_child(move_class, move_type, move_index)

    def get_children(self):
        """Generate all valid child states following the move constraints."""
        children = []
        for (move_class, move_type, move_index) in self.get_allowed_moves():
            children.append(self.get_child(move_class, move_type, move_index))
        return children

    def get_child_dict(self):
        """Generate a dictionary with all child constraints. Does not consider move constraints."""
        children = {}
        for move_class in range(len(CubeState.moves)):
            for move_type in range(len(CubeState.moves[move_class])):
                for move_index in range(len(CubeState.moves[move_class][move_type])):
                    move = CubeState.moves[move_class][move_type][move_index]
                    children[move] = self.get_child(move_class, move_type, move_index)
        return children

