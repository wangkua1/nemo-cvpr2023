import numpy as np


def find_first_index(arr, v):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    res = np.where(arr == v)[0]
    if len(res) == 0:
        return -1
    else:
        return res[0]