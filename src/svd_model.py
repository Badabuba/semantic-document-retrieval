import numpy as np

def manual_svd(A_matrix, k_dimensions=100, max_iters=100, tol=1e-5):
    """
    Ручна реалізація Truncated SVD за допомогою Power Iteration.
    """
    print(f"Запуск ручного SVD для k={k_dimensions}...")

    n_docs, n_terms = A_matrix.shape

    U = np.zeros((n_docs, k_dimensions))
    S = np.zeros(k_dimensions)
    Vt = np.zeros((k_dimensions, n_terms))

    print("Обчислення матриці коваріації A^T * A...")
    ATA = A_matrix.T @ A_matrix

    for i in range(k_dimensions):
        v = np.random.rand(n_terms)
        v = v / np.linalg.norm(v)

        for _ in range(max_iters):
            v_new = ATA @ v

            for j in range(i):
                v_new -= np.dot(Vt[j], v_new) * Vt[j]

            v_new = v_new / np.linalg.norm(v_new)

            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        Av = A_matrix @ v
        sigma = np.linalg.norm(Av)

        if sigma > 1e-10:
            u = Av / sigma
        else:
            u = np.zeros(n_docs)

        Vt[i, :] = v
        S[i] = sigma
        U[:, i] = u

        if (i + 1) % 10 == 0:
            print(f"Знайдено {i + 1}/{k_dimensions} компонентів...")

    print("Проєкція документів у новий простір...")
    d_new_matrix = U * S

    class ManualSVDModel:
        def __init__(self, Vt):
            self.components_ = Vt

        def transform(self, X):
            return X @ self.components_.T

    return ManualSVDModel(Vt), d_new_matrix

def apply_svd(A_matrix, k_dimensions=100):
    return manual_svd(A_matrix, k_dimensions)
