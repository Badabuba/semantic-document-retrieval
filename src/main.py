from data_prep import get_stopwords, load_corpus, build_term_document_matrix
from svd_model import perform_manual_svd, truncate_svd
from search import project_query, rank_documents
import numpy as np

def run_lsa_engine(documents: list, k: int, query: str):
    """
    Orchestrates the LSA process for a given set of documents and a query.
    """
    stops = get_stopwords()

    # Matrix A (Manual TF-IDF)
    A_list, vocab, idfs = build_term_document_matrix(documents, stops)

    # Perform Manual SVD & Truncation
    U, S, VT = perform_manual_svd(A_list)
    U_k, S_k, VT_k = truncate_svd(U, S, VT, k)

    # Create Document Semantic Map (D_sem = S_k * VT_k)
    D_sem = np.dot(S_k, VT_k)

    # Project query
    q_sem = project_query(query, vocab, idfs, U_k, stops)

    # Ranking
    results_idx, scores = rank_documents(q_sem, D_sem)

    # Output Display
    print(f"\nQuery: '{query}' (k={k})")
    for idx in results_idx:
        print(f"Score: {scores[idx]:.4f} | Doc: {documents[idx]}")

if __name__ == "__main__":
    # my_docs = load_corpus("my_database.txt")

    my_docs = [
    "The feline purred on the couch.", # Cat concept
    "A small kitten is playing with yarn.", # Cat concept
    "The kitten sat on the couch.", # Cat concept
    "The pilot of the plane told us to stop.", # Other concept
    "Friday evenings are the best." # Other concept
    ]
    users_query = "kitten on the mat"


    run_lsa_engine(my_docs, 2, users_query)
