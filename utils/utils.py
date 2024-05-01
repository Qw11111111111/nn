import numpy as np

def argwhere(v: list | np.ndarray, val: str | float | int) -> int:
    for i, value in enumerate(v):
        if str(value) == str(val):
            return i