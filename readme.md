# High-Performance Naive Bayes Classifier

## Overview
This repository contains a high-throughput Multinomial Naive Bayes inference engine designed for text classification on the 20 Newsgroups dataset.

Unlike standard iterative implementations, this project utilizes NumPy vectorization and broadcasting to perform training and inference as dense matrix operations, reducing CPU overhead and execution time.

## Technical Implementation
The core logic replaces Python loops with linear algebra primitives to ensure scalability:

- **Vectorized Inference:** Computes posterior probabilities using matrix multiplication rather than iterative lookups.
- **Log-Space Arithmetic:** Implements log-likelihood accumulation to prevent floating-point underflow on high-dimensional sparse vectors.
- **O(1) Class Lookup:** Uses optimized index mapping for constant-time label retrieval during batch prediction.

## Performance Benchmarks
Tested on the 20 Newsgroups dataset (approx. 18,000 documents, 20 classes).

## Due to repository size limits, the **20 Newsgroups** dataset is not included in the source tree. 

1. Download the dataset from the [original archive](https://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz).
2. Extract the contents.
3. Rename the folder to `20_newsgroups` and place it in the project root.

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Accuracy** | 83.49% | Matches Scikit-Learn baseline |
| **Vocabulary Size** | ~17,900 | Filtered (min_freq=10) |
| **Execution Time** | ~47s | End-to-end (Load + Train + Inference) | 

## Usage

### Prerequisites
- Python 3.8+
- NumPy

### Installation
pip install -r requirements.txt

### Execution
python NBClassifier.py

## Project Structure
- NBClassifier.py: The vectorized implementation
- 20_newsgroups/: Dataset directory
- requirements.txt: Dependencies
- README.md: Documentation
