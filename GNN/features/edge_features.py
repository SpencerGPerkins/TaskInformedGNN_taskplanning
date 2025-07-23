import torch

def build_edge_index_adj_matrix(num_wires, num_terminals=1):
    edge_index = []

    for i in range(num_wires):
        for j in range(num_terminals):
            edge_index.append([i, num_wires + j])
            edge_index.append([num_wires + j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    adj_matrix = torch.zeros(num_wires + num_terminals, num_wires + num_terminals)
    print(edge_index)
    for src, dst in edge_index.t():
        adj_matrix[src, dst] = 1 

    return edge_index, adj_matrix

def edge_feature_encoding(wire_list, terminal_dict, distance_function, target_wire_color):
    edge_features = []
    distances = []
    num_wires, num_terminals = len(wire_list), len(terminal_dict)

    for i in range(num_wires):
        # dist = distance_function(torch.tensor(wire_list[i]["normalized_coordinates"]), torch.tensor(terminal_dict["normalized_coordinates"]))
        dist = distance_function(torch.tensor(wire_list[i]["coordinates"]), torch.tensor(terminal_dict["coordinates"]))
        distances.append(dist)
    
    min_dist = min(distances)
    max_dist = max(distances)
    denom = max_dist - min_dist if max_dist > min_dist else 1e-6

    normed = [1- ((d - min_dist) / denom) for d in distances]

    # Distance and relevance weight
    for v, val in enumerate(normed):
        # edge_features.append([val]) # Wire to Terminal
        if wire_list[v]["color"] == target_wire_color:
            edge_features.append([val, 5.]) 
            edge_features.append([val, 5.])
        else:
            edge_features.append([val, 0.])
            edge_features.append([val, 0.])
        # edge_features.append([val]) # Terminal to Wire
    print("IN THE EDGE FEATURE FUNCTION\n")
    print(f"normed: {normed}")
    print(f"distances: {distances}")
    
    return torch.tensor(edge_features, dtype=torch.float), distances