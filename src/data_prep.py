import math
import re
import nltk
from nltk.corpus import stopwords

def get_stopwords():
    """
    Ensures stopwords are available and returns them as a set
    """
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))


def clean_text(text: str, stop_words: set) -> list:
    """
    Lowercase, remove punctuation, and filter out provided stop-words
    """
    text = text.lower()
    words = re.findall(r'\b[a-z]{2,}\b', text)
    cleaned = [w for w in words if w not in stop_words and not w.isdigit()]
    return cleaned

def build_term_document_matrix(documents: list, stop_words: set):
    """
    Manually calculates TF-IDF.
    """
    # Preprocessing
    processed_docs = [clean_text(doc, stop_words) for doc in documents]
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]

    # Build Vocabulary
    vocabulary = sorted(list(set(word for doc in processed_docs for word in doc)))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    num_docs = len(processed_docs)
    num_words = len(vocabulary)

    # Matrix A
    A = [[0.0 for _ in range(num_docs)] for _ in range(num_words)]
    
    # Calculate Document Frequency (df)
    df = [0] * num_words
    for doc in processed_docs:
        unique_words_in_doc = set(doc)
        for word in unique_words_in_doc:
            if word in word_to_index:
                df[word_to_index[word]] += 1

    # Calculate TF-IDF Weights
    for j, doc in enumerate(processed_docs):
        counts = {}
        for word in doc:
            counts[word] = counts.get(word, 0) + 1

        for word, count in counts.items():
            i = word_to_index[word]

            # TF
            tf = count / len(doc)

            # IDF
            idf = math.log(num_docs / df[i])

            A[i][j] = tf * idf

    return A, vocabulary

def load_corpus(file_path):
    """
    Assumes each line in a text file is a 'document'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


if __name__ == "__main__":
    #Test
    stops = get_stopwords()

    sample_docs = [
        "The cat sat on the mat.", 
        "The dog sat on the log.",
        "The bird flew over the log."
    ]
    
    matrix, vocab = build_term_document_matrix(sample_docs, stops)

    print(f"Vocabulary ({len(vocab)} words): {vocab}")
    print(f"Matrix shape: {len(matrix)} rows (terms) x {len(matrix[0])} columns (docs)")

    n = 3
    print(f"\nSample of Matrix A (first {n} rows):")
    for idx, row in enumerate(matrix[:n]):
        formatted_row = [round(val, 4) for val in row]
        print(f"Word '{vocab[idx]}': {formatted_row}")

