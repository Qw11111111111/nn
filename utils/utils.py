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
        

def timeit(func: object) -> None:

    def inner(*args, **kwargs) -> None:
        start = perf_counter()
        returned_values = func(*args, **kwargs)
        end = perf_counter()
        print(f"time taken by function {func.__name__}: {end - start}")
        return returned_values
    
    return inner