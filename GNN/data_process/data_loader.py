import torch
from torch_geometric.data import Data, DataLoader
from ..graph import InferenceTaskGraphHeterogeneous


def load_dataset(vision_path, llm_path, label_path, num_data_samples: int, action_primitives: list):
    data_list = [] 
    for d in range(num_data_samples):
        g_id = d
        print(f"Graph ID: {d}")
        vision_data = f"{vision_path}sample_{d}.json"
        llm_data = f"{llm_path}sample_{d}.json"
        label_data = f"{label_path}sample_{d}.json"

        G = InferenceTaskGraphHeterogeneous(
            action_primitives = action_primitives,
            vision_path = vision_data,
            llm_path=llm_data,
            label_path=label_data
        )

        wire_encodings, terminal_encodings = G.get_node_features()
        x = torch.cat([wire_encodings, terminal_encodings.unsqueeze(0)])

        edge_index = G.get_edge_index()
        edge_attr = G.get_edge_attr()
        y_wire_global, y_wire_local, y_action = G.get_labels()

        wire_mask, terminal_mask = G.get_node_masks() 
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr= edge_attr,
            y_wire_global=y_wire_global.unsqueeze(0),
            y_wire_local=y_wire_local.unsqueeze(0),
            y_action=y_action.unsqueeze(0),
            wire_mask=wire_mask,
            terminal_mask=terminal_mask,
            graph_id=g_id
        )
                        )   
        print(f"Processing Data: {d}")

    return data_list