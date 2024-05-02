import numpy as np
from time import perf_counter

def argwhere(v: list | np.ndarray, val: str | float | int, axis: int = 0) -> int | tuple[int]:
    if axis == 0:
        for i, value in enumerate(v):
            if str(value) == str(val):
                return i
    elif axis == 1:
        for i, row in enumerate(v):
            for j, value in enumerate(row):
                if str(value) == str(val):
                    return (i, j)
        

def timeit(func):
    pass