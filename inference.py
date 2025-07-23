import torch
import GNN

import config
from torch_geometric.data import Data
import networkx as nx
import json

def run_inference():
    # Load model 
    device = config.DEVICE
    possible_actions = config.ACTION_PRIMS

    # Save data for history and next action
    data_saver = {
        "predicted_wire": [],
        "predicted_action": [],
        "previous_actions": []
    }

    with open("run_data/action_executions.json", "r") as in_file:
        history = json.load(in_file)
    data_saver["previous_actions"] = history["previous_actions"]

    # Load a new sample
    vision_data = f"run_data/vision/vision_to_gnn.json"
    llm_data = f"run_data/llm/llm_to_gnn.json"

    # Create graph
    graph = GNN.graph.InferenceTaskGraphHeterogeneous(possible_actions, vision_data, llm_data)
    
    wire_encodings, terminal_encodings = graph.get_node_features()
    x = torch.cat([wire_encodings, terminal_encodings.unsqueeze(0)])

    edge_index = graph.get_edge_index()
    edge_attr = graph.get_edge_attr()
    wire_mask, terminal_mask = graph.get_node_masks()     
    wire_map = graph.get_wire_mappings()

    data = Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                wire_mask=wire_mask,
                terminal_mask=terminal_mask,
                )
    data.to(device)

    model_params = torch.load(config.MODEL_PARAMS_PATH, map_location=device)
    model = GNN.models.TwoHeadGATSmall(
        in_dim=data.x.shape[1],
        edge_feat_dim=2,
        hidden_dim=config.HIDDEN_DIM,
        num_actions=len(possible_actions),      
    )

    with torch.no_grad():
        model.load_state_dict(model_params)
        model.to(device)
        model.eval()

        wire_logits, action_logits = model(
            data.x.float(), wire_mask, data.edge_index, data.edge_attr, data.batch
        )

        predicted_wire = wire_logits.argmax(dim=0).cpu().numpy().tolist()
        predicted_action = action_logits.argmax(dim=1).cpu().numpy().tolist()


        data_saver["predicted_action"] = [possible_actions[predicted_action[-1]]]
        data_saver["previous_actions"].append(possible_actions[predicted_action[-1]])
        print(f"Predicted Wire Index: {predicted_wire}")
        print(f"Predicted Action: {possible_actions[predicted_action[0]]}")

sample_label = "../task_plan/working_dir/dataset/generated_synthetic_dataset_0714/labels/sample_500.json"

with open(sample_label, "r") as label_read:
    file = json.load(label_read)
run_inference()
print(f"Action Label: {file['correct_action']}")
print(f"GLOBAL Wire ID: {file['target_wire']['ID']}")


        
