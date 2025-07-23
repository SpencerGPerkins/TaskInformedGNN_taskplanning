import os

def load_checkpoint(checkpoint_path, model, load_function):
        checkpoint = load_function(checkpoint_path, map_location="cpu")
        model_dict = model.state_dict()    
        pretrained_dict = {k:v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"\n\nCheckpoint Loaded at {checkpoint_path}")

        return model
