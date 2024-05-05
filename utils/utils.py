import numpy as np
from time import perf_counter
from treelib import Node, Tree

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
        print(f"time taken by function {func.__name__}: {end - start:.3f}")
        return returned_values
    
    return inner

def plot_tree(data: list[int]) -> None:
    tree = Tree()
    indices = [[],[]]
    for i, item in enumerate(data):
        if i == 0:
            tree.create_node("Root","0_1")
            indices[0].append(item)
            continue
        #find the branch
        idx = argwhere(indices, )
        tree.create_node(f"{i}_1", f"{i}_1")
        tree.create_node(f"{i}_2", f"{i}_2")

    pass