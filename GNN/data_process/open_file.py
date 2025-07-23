import json
import os


def open_helper(file_path):
    """Opens JSON file."""
    print(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")
