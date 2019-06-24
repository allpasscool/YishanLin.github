import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])
    for i in range(0, M):
        for j in range(0, N):
            tmp = 0
            for k in range(0, K):
                tmp += (X[i, k] - Y[j, k]) ** 2
            D[i, j] = np.sqrt(tmp)
    return D

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])

    for i in range(0, M):
        for j in range(0, N):
            tmp = 0
            for k in range(0, K):
                tmp += abs(X[i, k] - Y[j, k])
            D[i, j] = tmp
    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])

    for i in range(0, M):
        for j in range(0, N):
            tmp = 0
            tmp1 = 0
            tmp2 = 0
            for k in range(0, K):
                tmp += (X[i, k] * Y[j, k])
                tmp1 += X[i, k] * X[i, k]
                tmp2 += Y[j, k] * Y[j, k]
                
            tmp1 = np.sqrt(tmp1)
            tmp2 = np.sqrt(tmp2)
            D[i, j] = 1 - (tmp / (tmp1 * tmp2))
            #D[i, j] = tmp / tmp1
    return D