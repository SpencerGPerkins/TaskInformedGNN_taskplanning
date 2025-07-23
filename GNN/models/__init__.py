# models/__init__.py
import torch
from .TwoHeadGAT import TwoHeadGAT
from .TwoHeadGAT_small import TwoHeadGATSmall

def load_model(weights_path: str, model_size: str, device="cpu"):
    if model_size == "small":
        model = TwoHeadGATSmall()
    elif model_size == "medium":
        model = TwoHeadGAT()
    else:
        raise ValueError(f"{model_size} is not a valid string for model sizes...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

__all__ = [
    "TwoHeadGAT",
    "TwoHeadGATSmall",
    "load_model"
]