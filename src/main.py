# https://www.kaggle.com/datasets/au1206/20-newsgroup-original

from data_prep import get_stopwords, load_corpus, build_term_document_matrix
from svd_model import perform_manual_svd, truncate_svd
from search import project_query, rank_documents
import numpy as np

def run_lsa_engine(documents: list, k: int, query: str, top_n: int, result_f: str):
    """
    Orchestrates the LSA process for a given set of documents and a query.
    -documents: list of database documents
    -query: user`s query
    -top_n: how many documents to output
    -result_f: where to store results
    """
    stops = get_stopwords()

    # Matrix A (Manual TF-IDF)
    A, vocab, idfs = build_term_document_matrix(documents, stops)

    # Perform Manual SVD, Truncation
    print(f"Starting SVD...")
    U, S, VT = perform_manual_svd(A)
    print(f"SVD done")
    U_k, S_k, VT_k = truncate_svd(U, S, VT, k)

    # Create Document Semantic Map (D_sem = S_k * VT_k)
    D_sem = np.dot(S_k, VT_k)

    # Project query
    q_sem = project_query(query, vocab, idfs, U_k, stops)

    # Ranking
    results_idx, scores = rank_documents(q_sem, D_sem)

    # write results
    top_indices = results_idx[:top_n]

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"LSA SEARCH ENGINE RESULTS\n")
        f.write(f"Query: {query}\n")
        f.write(f"Parameters: k={k}, docs={len(documents)}\n")
        f.write("="*50 + "\n\n")

        for i in range(top_n):
            doc_idx = top_indices[i]
            score = scores[doc_idx]
            content = documents[doc_idx]

            f.write(f"RESULT #{i+1}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Document Text:\n{content}\n")
            f.write("-" * 30 + "\n\n")

    print(f"Saved the top {top_n} results to 'results.txt'")

if __name__ == "__main__":
    # UNCOMMENT THIS TO TRY ON A CUSTOM DATASET AND COMMENT ALL CODE BELOW
    test_documents = [
        # Cluster 1: Space & Astronomy (Shared words: Space, NASA, Mission, Orbit)
        "NASA launched a new space mission to study the orbit of distant planets.",
        "The space telescope is part of a NASA mission to photograph deep space galaxies.",
        "Astronauts on the space station are monitoring the orbit of the satellite.",
        "A mission to Mars is being planned by NASA using advanced space technology.",
        "The moon is in orbit around Earth and is a target for future space exploration.",

        # Cluster 2: Culinary Arts (Shared words: Recipe, Cooking, Chef, Dish)
        "The chef shared a secret recipe for a classic Italian pasta dish.",
        "Cooking a gourmet dish requires following a precise recipe from a professional chef.",
        "This recipe for roasted chicken is a favorite dish among home cooking enthusiasts.",
        "The restaurant chef specializes in a spicy dish using a traditional cooking recipe.",
        "Every chef knows that a great dish starts with fresh ingredients and a good recipe.",

        # Cluster 3: Programming (Shared words: Code, Software, Program, Language)
        "Software developers write code using the Python programming language.",
        "This computer program uses efficient code to improve software performance.",
        "Learning a new programming language is essential for modern software development.",
        "The software engineer debugged the code to fix a bug in the program.",
        "A clean program requires well-documented code and solid software architecture."
    ]

    users_query = "Cooking"
    k = 5
    top_documents = 15
    run_lsa_engine(test_documents, k, users_query, top_documents, "result.txt")


