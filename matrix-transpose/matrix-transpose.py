import numpy as np

def matrix_transpose(A):
    A = np.asarray(A)
    N, M = A.shape
    B = np.empty((M, N), dtype=A.dtype)
    
    # Build index grids for every (i, j) position in A
    rows, cols = np.indices((N, M))   # rows, cols both shape (N, M)
    
    # Swap the coordinates when writing into B
    B[cols, rows] = A[rows, cols]
    
    return B