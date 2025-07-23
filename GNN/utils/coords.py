import numpy as np

def match_coords(a, b, tol=1e-3):
    """Match coordinates using np.round and np.allclose
    Params:
    ------
        a: list, first coordinate (xyz)
        b: list, second coordinate
        tol: int, tolerance for np.allclose

    Returns:
    -------
        bool: True if coordinates match within tolerance
    """
    try:
        a_arr = np.round(np.array(a, dtype=np.float64), decimals=4)
        b_arr = np.round(np.array(b, dtype=np.float64), decimals=4)

        return np.allclose(a_arr, b_arr, atol=tol)
    except Exception as e:
        print(f"Coord matching failed for {a} vs {b} — {e}")

        return False  
    
def normalize(values):   
    """Normalize coordinate arrays
    Params:
    -------
        values: ndarray, Shape (N, D) array of coordinates

    Returns:
    -------
        ndarray: Normalized coordinates
    """
    # Compute normalization stats
    mean = values.mean(axis=0)
    std = values.std(axis=0) + 1e-8  
    # Normalize all coords
    norm_coords = (values - mean) / std

    return norm_coords