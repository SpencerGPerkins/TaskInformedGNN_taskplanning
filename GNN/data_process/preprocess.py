def parse_target_info(llm_data, color_list):
    """ Parse the LLM data (json file)"""
    color, _ = llm_data["target_wire"].split("_")
    _, terminal_number = llm_data["target_terminal"].split("_")

    return {
        "wire_color": color,
        "wire_color_idx": color_list.index(color),  # Index of target wire's color within self.colors
        "terminal_id": int(terminal_number),
        "terminal_name": llm_data["target_terminal"],
        "goal": llm_data["goal"]
    } 

def extract_wire_nodes(all_wires, target_info):
    """Extract target wires, defined by target wire color ----Real-time Inference Implementation-------

    Params:
    -------
        all_wires: list, each element is a dict corresponding to a detected wire
        target_info: dict, target information extracted from LLM data

    Returns:
    -------
        wires: list, detected wires that correspond to the target color from the LLM
    """
    wires = []
    for idx, wire in enumerate(all_wires): # Iterate through detected wires for target wires based on color
        if wire["color"] == target_info["wire_color"]:
            dict_entry = {
                "id": idx,
                "color": wire["color"],
                "state": wire["state"],
                "coordinates": wire["position"]
            }
            wires.append(dict_entry)
            print(f"Wire {idx} processed...")
        else:
            continue    
    
    return wires

def extract_wire_nodes_statebased(all_wires, target_info):
    """Extract target wires, defined by target wire color, and check for other manipulated wires
    Params:
    -------
        all_wires: list, each element is a dict corresponding to a detected wire
        target_info: dict, target information extracted from LLM data

    Returns:
    -------
        wires: list, detected wires that correspond to the target color from the LLM
    """
    wires = []
    wire_mapping = []
    local_id = 0
    for idx, wire in enumerate(all_wires): # Iterate through detected wires for target wires based on color
        if wire["color"] == target_info["wire_color"]:
            dict_entry = {
                "id": idx,
                "color": wire["color"],
                "state": wire["state"],
                "coordinates": wire["position"]
            }

            mapping_entry = {
                "global_id": idx,
                "local_id": local_id,
                "coordinates": wire["position"]
            }

            wires.append(dict_entry)
            wire_mapping.append(mapping_entry)
            print(f"Wire {idx} processed...")
            
        elif wire["state"] == "held":
            dict_entry = {
                "id": idx,
                "color": wire["color"],
                "state": wire["state"],
                "coordinates": wire["position"]
            }

            mapping_entry = {
                "global_id": idx,
                "local_id": local_id,
                "coordinates": wire["position"]
            }

            wires.append(dict_entry)
            wire_mapping.append(mapping_entry)
            print(f"Wire {idx} processed...")

        else:
            continue    

    return wires, wire_mapping

def extract_terminal_node(all_terminals, target_info):
    """Extract the target terminal, defined by name from LLM
    Params:
    -------
        all_terminals: dict, all of the known (or detected) terminals
        target_info: dict, target information extracted from LLM data

    Returns:
    -------
        dict, the target terminal's information
    """
    terminal_name = target_info["terminal_name"]
    coords = all_terminals[terminal_name]["position"]
    state = all_terminals[terminal_name]["state"]

    return {
        "id": target_info["terminal_id"],
        "name": terminal_name,
        "state": state,
        "coordinates": coords
    }


def match_label_to_wire(label_data, wire_list, match_coords_fn, color_list):
    """Find the wire in wire_dict that matches the label coordinates
    Params:
    -------
        label_data: dict, contains label information for supervised learning
        wire_list: list, all wires that are of target color
        match_coords_fn: function, used to match the wire coordinates and target_wire_coords (find the target wire)
        color_list: list, strings of possible colors for wires

    Returns:
    --------
        dict, labels for supervised training 
    """
    label_wire_color, _ = label_data["target_wire"]["name"].split("_")
    label_wire_coords = label_data["target_wire"]["position"]

    matched_wire = next(
        (wire for wire in wire_list if match_coords_fn(wire["coordinates"], label_wire_coords)),
        None
    )

    if matched_wire is None:
        raise ValueError(f"No wire matched label coordinates: {label_wire_coords}")   
    
    return {
        "wire_color": label_wire_color,
        "wire_color_idx": color_list.index(label_wire_color),
        "wire_coordinates": label_data["target_wire"]["position"],
        "global_wire_id": matched_wire["id"],
        "local_wire_id": None,

        "terminal_id": int(label_data["target_terminal"]["name"].split("_")[1]),
        "terminal_coordinates": label_data["target_terminal"]["position"],
        
        "action": label_data["correct_action"],
        "action_one_hot": None # Set later
    }