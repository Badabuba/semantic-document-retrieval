# Semantic Document Retrieval Engine (LSA & SVD)

This repository contains an automated Information Retrieval system built from scratch. Unlike standard keyword-based search engines, this system utilizes Latent Semantic Analysis (LSA) and Singular Value Decomposition (SVD) to "read between the lines" and retrieve documents based on their underlying conceptual meaning rather than literal surface-level character matching.

## 👥 Team & My Role
* **Team:** Collaborative university project (3 students).
* **My Core Contributions:**
  * Collaborated on the mathematical architecture, helping translate linear algebra concepts (TF-IDF, SVD, Power Iteration) into the system's logic.
  * Conducted system evaluation and hyperparameter tuning, identifying k=50 as the optimal dimensionality bottleneck to capture ~64.8% of corpus variance while filtering out linguistic noise.
  * Assisted in defining the retrieval logic and evaluating Cosine Similarity metrics against the BBC News dataset to validate the model's semantic accuracy.

## 🚀 Key Engineering Features
* **Custom SVD Implementation:** Bypassed heavy external machine learning libraries by implementing Singular Value Decomposition from scratch using Power Iteration and Hotelling's Deflation.
* **Semantic Concept Space:** Truncates the high-dimensional Term-Document matrix into a dense k-rank approximation, allowing the system to successfully handle synonymy and polysemy (e.g., matching "football forward" to "striker" despite zero keyword overlap).
* **Optimized Execution:** The custom SVD calculates only the top-k components directly, reducing time complexity from O(n³) to O(k * n²) and significantly reducing CPU and thermal overhead.

## 🛠️ Tech Stack & Architecture
* **Language:** Python
* **Libraries:** NumPy (high-performance matrix operations), Pandas (dataset management)
* **Dataset:** BBC News Corpus (2,225 documents across 5 thematic categories)

### Module Breakdown
* data_prep.py: Transforms the raw corpus into a Term-Document Matrix via tokenization, stop-word removal, and TF-IDF weighting.
* svd_model.py: The mathematical core. Executes Power Iteration to directly compute and output the low-rank truncated matrices Uk, Σk, and Vkᵀ.
* search.py: Projects raw user queries into the concept space and calculates Cosine Similarity scores against the document vectors.
* main.py: The execution orchestrator allowing dynamic control over hyperparameters.

---

<details>
  <summary><b>⚙️ Click to expand: Usage & Quick Start</b></summary>

To interact with the engine, modify the execution block in main.py:

    if __name__ == "__main__":
        # 1. Define your query
        user_query = "football coach"
        
        # 2. Set semantic depth (k-dimensions)
        k = 50 
        
        # 3. Define number of results
        top_n = 5

*Results are automatically logged to results.txt for reproducibility, recording the query, k-value, and Cosine Similarity scores.*
</details>

<details>
  <summary><b>📊 Click to expand: Mathematical Underpinnings & Benchmarking</b></summary>

### 1. TF-IDF & Matrix Construction
The corpus is represented as a Term-Document Matrix where each entry is weighted using Term Frequency-Inverse Document Frequency (TF-IDF) to penalize filler words and highlight distinct topics.

### 2. Low-Rank Approximation
The system actively halts the SVD loop after 'k' iterations to construct a low-rank approximation. This projects massive, sparse matrices into dense, lower-dimensional representations, effectively filtering out stylistic noise and typographical errors.

### 3. Hyperparameter Tuning (The Optimal 'k')
During evaluation on the BBC dataset, we analyzed the decay of singular values generated during SVD. 
* At k=50, the system captures ~64.8% of the total variance of the entire dataset.
* This configuration preserves the core semantic relationships of the corpus while successfully discarding the remaining 35.2% of the variance, which is dominated by random word-co-occurrence noise.
* A cosine similarity score > 0.70 typically indicates a highly successful thematic match.
</details>
