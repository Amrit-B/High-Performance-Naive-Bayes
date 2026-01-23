# Naive Bayes Text Classifier
**ML Project 2 - Masters Sem 4**

## What This Project Does

This is a text classification system that automatically categorizes news articles into 20 different topics (sports, technology, politics, religion, etc.) using the **Multinomial Naive Bayes** algorithm.

The classifier learns patterns from training articles and then predicts categories for new articles based on word frequencies.

## How It Works

1. **Training Phase:** 
   - Reads 500 articles from each of the 20 categories
   - Counts word frequencies in each category
   - Calculates probabilities using Bayes theorem

2. **Testing Phase:**
   - Takes new articles it hasn't seen before
   - Compares word patterns to what it learned
   - Predicts the most likely category

3. **Key Features:**
   - Uses NumPy for efficient matrix operations
   - Implements log probabilities to avoid numerical underflow
   - Filters vocabulary to only include frequently occurring words

## Dataset Setup

This project uses the **20 Newsgroups** dataset, which is NOT included in this repo.

**To get the dataset:**
1. Download: [20 Newsgroups dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz)
2. Extract the archive
3. Make sure the extracted folder is named `20_newsgroups`
4. Place it in the project root directory

## Installation & Running

**Requirements:**
- Python 3.8+
- NumPy

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the classifier:**
```bash
python main.py
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | ~83% |
| Vocabulary Size | ~17,900 words |
| Training Set | 10,000 documents (500 per class) |
| Test Set | ~8,828 documents |
| Execution Time | ~47 seconds |

## Project Structure

```
├── main.py              # Main execution script with configuration
├── naive_bayes.py       # MultinomialNB classifier implementation
├── data_loader.py       # Data loading and tokenization functions
├── requirements.txt     # Python dependencies
├── NBClassifier.py      # Original single-file implementation
└── 20_newsgroups/       # Dataset directory (not included)
```

## What I Learned

- How Naive Bayes works for text classification
- The importance of log probabilities for numerical stability
- Using NumPy for efficient vectorized operations
- Handling text encoding issues (latin-1 for older datasets)
- Balancing code modularity vs simplicity
- Why sorting file lists matters for reproducibility

## Notes

- The "naive" assumption treats words as independent, which isn't technically true but works well in practice
- Laplace smoothing (+1 to all counts) prevents zero probabilities
- Words appearing less than 10 times are filtered out to reduce noise
- You can adjust `VOCAB_MIN_FREQUENCY` in main.py to experiment with different vocabulary sizes
