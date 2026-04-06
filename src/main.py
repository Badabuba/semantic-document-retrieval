from data_prep import load_and_preprocess_data
from svd_model import apply_svd
from search import search_query

def main():
    documents, vectorizer, A_matrix = load_and_preprocess_data()

    k = 300
    svd_model, d_new_matrix = apply_svd(A_matrix, k_dimensions=k)

    print("\nSystem is ready")
    
    # Запит для семантичної перевірки (згадується у Фазі 5 вашого звіту про NASA/space)
    test_query = "space exploration and satellites"
    search_query(test_query, vectorizer, svd_model, d_new_matrix, documents, top_n=3)

if __name__ == "__main__":
    main()