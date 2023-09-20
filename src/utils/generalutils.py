import math


def getRowsCols(num_cells: int, optim_cols = 2):
    if num_cells <= optim_cols:
        return 1, num_cells

    nrows = math.ceil(num_cells / optim_cols)
    return nrows, optim_cols