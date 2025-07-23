import numpy as np
import torch
from ..utils.one_hot import one_hot_encode

def build_node_features(wire_list, terminal_dict, pos_encoder):
    """ Features for Model 1: Positional encoding based / task specific graph
    Params:
    -------
    wire_list: list, all relevant detected wires (of target color/class)
    terminal_dict: dict, target terminal information
    pos_encoder: module, used for positional encoding on xyz coords
    label_info: dict, label information corresponding to correct wire (by ID), terminal, and action

    Returns:
    --------
    X_wires: list, node feature vectors for wires
    X_terminals: list, node feature vectors for terminals
    local_wire_id: int, the local ID of the target wire (ID within task specific wires)

    """

    X_wires = []
    # wire_coords = [np.array(wire["normalized_coordinates"]) for wire in wire_list]
    # coords_list = wire_coords + [terminal_dict["normalized_coordinates"]]
    wire_coords = [np.array(wire["coordinates"]) for wire in wire_list]
    coords_list = wire_coords + [terminal_dict["coordinates"]]
    coords = np.stack(coords_list) # Should be shape: (batchsize=1, N, 2)
    coords = coords[np.newaxis, :, :] # Maintain batch size axis for now in case batching 
    positions = pos_encoder(torch.tensor(coords))
    wire_positions = positions[:,:len(wire_list)].squeeze()
    terminal_positions = positions[:,len(wire_list):].squeeze()

    for w, wire in enumerate(wire_list):
        if wire_positions.dim() == 1:
            f = [1., 0.]
            f.extend(wire_positions.tolist())
        else:
            f = [1., 0.]
            f.extend(wire_positions[w,:].tolist())
        X_wires.append(f)        
    X_wires = X_wires
    
    # Terminal features
    f_term = [0., 1.]
    f_term.extend(terminal_positions.tolist())
    X_terminals = f_term

    return X_wires, X_terminals

def StateBased_build_node_features(wire_list, terminal_dict):
    possible_states = ["on_table", "held", "inserted", "empty", "locked"]
    X_wires = []
    for w, wire in enumerate(wire_list):
        print(wire)
        f = [1., 0.]
        state_encoding = one_hot_encode(wire["state"], possible_states)
        f.extend(state_encoding)

        X_wires.append(f)
    X_wire = X_wires

    f_term = [0., 1.]
    term_state_encoding = one_hot_encode(terminal_dict["state"], possible_states)
    f_term.extend(term_state_encoding)
    X_terminals = f_term

    return X_wires, X_terminals
    
       