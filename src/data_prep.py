from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data():
    print("Завантаження датасету 20 Newsgroups...")
    dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    documents = dataset.data

    print(f"Завантажено {len(documents)} документів.")

    print("Побудова матриці TF-IDF (Токенізація та видалення стоп-слів)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    A_matrix = vectorizer.fit_transform(documents)

    print(f"Розмірність матриці TF-IDF: {A_matrix.shape}")
    return documents, vectorizer, A_matrix
