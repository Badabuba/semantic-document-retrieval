import numpy as np

def manual_cosine_similarity(q_vec, D_matrix):
    dot_products = np.dot(D_matrix, q_vec)

    norm_q = np.sqrt(np.sum(q_vec ** 2))

    norms_d = np.sqrt(np.sum(D_matrix ** 2, axis=1))

    if norm_q == 0:
        norm_q = 1e-10
    norms_d[norms_d == 0] = 1e-10

    similarities = dot_products / (norm_q * norms_d)

    return similarities

def search_query(query, vectorizer, svd_model, d_new_matrix, original_documents, top_n=3):
    q_tfidf = vectorizer.transform([query])

    q_new = svd_model.transform(q_tfidf)

    q_vec = np.asarray(q_new).flatten()
    
    similarities = manual_cosine_similarity(q_vec, d_new_matrix)

    best_match_indices = similarities.argsort()[-top_n:][::-1]
    
    print(f"\n--- Результати пошуку для запиту: '{query}' ---")
    for i, idx in enumerate(best_match_indices):
        score = similarities[idx]
        preview = original_documents[idx][:200].replace('\n', ' ') 
        print(f"\n[Місце {i+1}] Подібність: {score:.4f}")
        print(f"Текст: {preview}...")