import numpy as np
from data_prep import clean_text

def project_query(query_text, vocabulary, idf_weights, U_k, stop_words):
    """
    Translates a text query into the semantic concept space using TF-IDF.
    """
    query_words = clean_text(query_text, stop_words)

    # Vectorize with TF-IDF
    q_tfidf = np.zeros(len(vocabulary))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    for word in query_words:
        if word in word_to_index:
            idx = word_to_index[word]
            # TF
            tf = query_words.count(word) / len(query_words)
            # IDF
            q_tfidf[idx] = tf * idf_weights[idx]

    # Project (q_sem = U_k^T * q_tfidf)
    q_sem = np.dot(U_k.T, q_tfidf)

    return q_sem

def get_cosine_similarity(vec1, vec2):
    """
    Calculates the cosine of the angle between two vectors
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def rank_documents(q_sem, D_sem):
    """
    Compares the query to all documents in the semantic space.
    D_sem is the matrix (Sigma_k * V^T_k) where each column is a doc.
    """
    num_docs = D_sem.shape[1]
    scores = []

    for j in range(num_docs):
        doc_vector = D_sem[:, j]
        score = get_cosine_similarity(q_sem, doc_vector)
        scores.append(score)

    scores = np.array(scores)
    print(scores[:10])
    ranked_indices = np.argsort(scores)[::-1]
    print(ranked_indices[:10])

    return ranked_indices, scores

