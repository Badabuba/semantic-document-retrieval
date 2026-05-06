from data_prep import get_stopwords, load_corpus, build_term_document_matrix
from svd_model import perform_manual_truncated_svd
from search import project_query, rank_documents
import numpy as np
import pandas as pd
import random
import textwrap

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
    U_k, S_k, VT_k = perform_manual_truncated_svd(A, k)
    print(f"SVD done")


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
            cleaned_content = " ".join(content.split())
            wrapped_content = textwrap.fill(cleaned_content, width=80)

            f.write(f"RESULT #{i+1}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Document Text:\n{wrapped_content}\n")
            f.write("-" * 30 + "\n\n")

    print(f"Saved the top {top_n} results to 'results.txt'")

if __name__ == "__main__":
    # this dataset consist of 5 topics: business, entertainment, politics, sport, tech
    dataset_path = "data/bbc-news-data.csv"

    try:
        print(f"Loading dataset from {dataset_path}...")

        df = pd.read_csv(dataset_path, sep='\t', on_bad_lines='skip')
        raw_documents = df['content'].dropna().astype(str).tolist()

        random.seed(42)
        random.shuffle(raw_documents)

        # take any number of documents (max = 2225)
        test_documents = raw_documents
        print(f"Loaded {len(test_documents)} documents.")

        # Set your parameters
        users_query = "football coach"
        k = 50  # 50 preserves 65% of variance, 70 preserves 80% of variance, look at plot.png
        top_documents = 10

        run_lsa_engine(test_documents, k, users_query, top_documents, "result.txt")

    except FileNotFoundError:
        print(f"{dataset_path} not found")
    except Exception as e:
        print(f"An error occurred: {e}")
