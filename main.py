import time
from data_loader import load_data
from naive_bayes import MultinomialNB

DATASET_PATH = r"20_newsgroups"
TRAIN_TEST_SPLIT = 500
VOCAB_MIN_FREQUENCY = 10

def main():
    print("Starting Naive Bayes Classifier...")
    start_time = time.time()


    print(f"Loading data from {DATASET_PATH}...")
    train_data, test_data, classes = load_data(DATASET_PATH, TRAIN_TEST_SPLIT)
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    print(f"Number of classes: {len(classes)}")


    print("\nTraining the model...")
    model = MultinomialNB(min_freq=VOCAB_MIN_FREQUENCY)
    model.train(train_data, classes)
    print(f"Vocabulary size: {len(model.vocab)} words")

    test_contents = [t[0] for t in test_data]
    test_labels = [t[1] for t in test_data]

    print("\nEvaluating on test set...")
    predictions = model.predict(test_contents)

    correct = sum(1 for p, a in zip(predictions, test_labels) if p == a)
    accuracy = (correct / len(test_data)) * 100

    print(f"\n{'_______________________________'}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(test_data)}")
    print(f"Time elapsed: {time.time() - start_time:.2f}s")
    print(f"{'_______________________________'}")


if __name__ == "__main__":
    main()