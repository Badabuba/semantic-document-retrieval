import numpy as np

def perform_manual_svd(matrix_A):
    A = np.array(matrix_A)
    m, n = A.shape
    k_max = min(m, n)

    # Find U
    aat = np.dot(A, A.T)
    evals_u, U = np.linalg.eigh(aat)

    # Sort and keep only the top k_max
    idx_u = evals_u.argsort()[::-1]
    U = U[:, idx_u][:, :k_max] 

    # Find V
    ata = np.dot(A.T, A)
    evals_v, V = np.linalg.eigh(ata)

    # Sort and keep only the top k_max
    idx_v = evals_v.argsort()[::-1]
    V = V[:, idx_v][:, :k_max]
    
    # Find Singular Values
    singular_values = np.sqrt(np.maximum(evals_u[idx_u][:k_max], 0))

    return U, singular_values, V.T

def truncate_svd(U, S, VT, k):
    """
    Reduces the dimensionality to k concepts
    """
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    return U_k, S_k, VT_k

if __name__ == "__main__":
    test_A = [[1, 0], [0, 1], [1, 1]]
    U, S, VT = perform_manual_svd(test_A)

    print("Singular Values:", S)
    print("U shape:", U.shape)
    print("V^T shape:", VT.shape)

    Uk, Sk, VTk = truncate_svd(U, S, VT, k=1)
    print("Truncated S shape:", Sk.shape)

