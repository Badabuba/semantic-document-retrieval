import numpy as np

def power_iteration_deflation(M, num_components, num_iters=1000, tol=1e-6):
    """
    Calculates eigenvalues and eigenvectors manually using Power Iteration.
    """
    n, _ = M.shape
    eigenvalues = np.zeros(num_components)
    eigenvectors = np.zeros((n, num_components))

    M_copy = np.copy(M)

    for i in range(num_components):
        np.random.seed(42 + i)
        # random normalised vector
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        for _ in range(num_iters):
            # new_v = M * v, normalise it after
            v_next = np.dot(M_copy, v)
            v_next = v_next / np.linalg.norm(v_next)

            # Check if they are the same or perfectly opposite
            diff_same = np.linalg.norm(v_next - v)
            diff_opp  = np.linalg.norm(v_next + v)
            if min(diff_same, diff_opp) < tol:
                v = v_next
                break

            v = v_next
            
        # lambda = v^T*M*v
        eigenvalue = np.dot(v.T, np.dot(M_copy, v))
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = v

        # M = M - lambda*v*v^T
        M_copy = M_copy - eigenvalue * np.outer(v, v)

    return eigenvalues, eigenvectors


def perform_manual_svd(matrix_A):
    A = np.array(matrix_A)
    m, n = A.shape
    k_max = min(m, n)

    # A^T*A
    ata = np.dot(A.T, A)

    # V and Eigenvalues
    evals_v, V = power_iteration_deflation(ata, k_max)

    # Singular Values
    singular_values = np.sqrt(np.maximum(evals_v, 0))
    print(singular_values)

    # U_i = (A * V_i) / Sigma_i
    U = np.zeros((m, k_max))
    for i in range(k_max):
        if singular_values[i] > 1e-10:
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(m)

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
    test_B = [[0.0, 0.0, 0.0, 0.0, 0.5364793041447], [0.3054302439580517, 0.0, 0.0, 0.3054302439580517, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5364793041447], [0.5364793041447, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5364793041447], [0.0, 0.22907268296853878, 0.0, 0.3054302439580517, 0.0], [0.0, 0.0, 0.3218875824868201, 0.0, 0.0], [0.0, 0.0, 0.3218875824868201, 0.0, 0.0], [0.0, 0.40235947810852507, 0.0, 0.0, 0.0], [0.5364793041447, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5364793041447, 0.0], [0.0, 0.40235947810852507, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3218875824868201, 0.0, 0.0], [0.0, 0.0, 0.3218875824868201, 0.0, 0.0], [0.0, 0.0, 0.3218875824868201, 0.0, 0.0], [0.0, 0.40235947810852507, 0.0, 0.0, 0.0]]
    U, S, VT = perform_manual_svd(test_B)

    print("Singular Values:", S)
    print("U shape:", U.shape)
    print("V^T shape:", VT.shape)

    Uk, Sk, VTk = truncate_svd(U, S, VT, k=3)
    print("Truncated S shape:", Sk.shape)

