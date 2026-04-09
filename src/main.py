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
    A_list, vocab, idfs = build_term_document_matrix(documents, stops)

    # Perform Manual SVD & Truncation
    print(f"Starting SVD...")
    U, S, VT = perform_manual_svd(A_list)
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
    top_scores = scores[:top_n]

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"LSA SEARCH ENGINE RESULTS\n")
        f.write(f"Query: {users_query}\n")
        f.write(f"Parameters: k={k}, docs={len(documents)}\n")
        f.write("="*50 + "\n\n")

        for i in range(top_n):
            doc_idx = top_indices[i]
            score = top_scores[i]
            content = documents[doc_idx]
            f.write(f"RESULT #{i+1}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Document Text:\n{content}\n")
            f.write("-" * 30 + "\n\n")

    print(f"Saved the top {top_n} results to 'results.txt'")

if __name__ == "__main__":
    # UNCOMMENT THIS TO TRY ON A CUSTOM DATASET AND COMMENT ALL CODE BELOW
    my_docs = [
    "The feline purred on the couch.", # Cat concept
    "A small kitten is playing with yarn.", # Cat concept
    "The pilot of the plane told us to stop.", # Other concept
    "The kitten sat on the couch.", # Cat concept
    "Friday evenings are the best." # Other concept
    ]
    users_query = "kitten on the mat"
    k = 2
    run_lsa_engine(my_docs, k, users_query, 2, "result.txt")


    # from sklearn.datasets import fetch_20newsgroups
    # print("Loading 20 Newsgroups data...")
    # newsgroups = fetch_20newsgroups(subset='train', 
    #                                 categories=['sci.space'],
    #                                 remove=('headers', 'footers', 'quotes'))

    # my_docs = [doc for doc in newsgroups.data if len(doc.strip()) > 100]

    # k = 20
    # users_query = "uranium"

    # print(f"Running LSA on {len(my_docs)} documents with k={k}...")
    # run_lsa_engine(my_docs, k, users_query, 3, "result.txt")
