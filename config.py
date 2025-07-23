# config.py
import torch

#=====General config==================================================#
# Action Primitives
ACTION_PRIMS = ["pick", "insert", "lock", "putdown"]

# Model params
MODEL_SIZE = "small"
HIDDEN_DIM = 64
NUM_ACTIONS = 4

GRAPH_TYPE = "task_specific" # Supports "task_specific", "task_agnostic"
NODE_FEATURE_TYPE = "state" # Supports "positional", "state"

MODEL_PARAMS_PATH = f"GNN/TwoHeadGAT_{MODEL_SIZE}_7172025_{NODE_FEATURE_TYPE}_{GRAPH_TYPE}.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")