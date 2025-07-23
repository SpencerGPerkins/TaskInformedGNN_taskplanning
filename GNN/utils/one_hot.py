def one_hot_encode(value, categories):
    """One-hot encoding for action labels
    Params:
    -------
        value: string, category for encoding
        categories: list, all possible categories (number of dims for encoding)

    Returns:
    -------
        encoding, list 1 if index of category else 0
    """
    
    encoding = [0] * len(categories)
    encoding[categories.index(value)] = 1

    return encoding