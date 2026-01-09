import os
import re
import time
import numpy as np

DATASET_PATH = r"20_newsgroups"
TRAIN_TEST_SPLIT = 500
VOCAB_MIN_FREQUENCY = 10

def load_data(dataset_dir):
    training_data = []
    testing_data = []
    
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        filenames = sorted([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])

        for idx, fname in enumerate(filenames):
            filepath = os.path.join(class_path, fname)
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
            
            if idx < TRAIN_TEST_SPLIT:
                training_data.append((content, class_name))
            else:
                testing_data.append((content, class_name))

    return training_data, testing_data, class_names

def tokenize(text):
    return re.findall(r'[a-zA-Z]+', text.lower())

class MultinomialNB:
    def __init__(self, min_freq=VOCAB_MIN_FREQUENCY):
        self.min_freq = min_freq
        self.vocab = {}
        self.class_map = {}
        self.index_map = {} 
        self.log_priors = None
        self.log_likelihoods = None

    def train(self, training_data, class_names):
        self.class_map = {name: i for i, name in enumerate(class_names)}
        self.index_map = {i: name for name, i in self.class_map.items()} 
        num_classes = len(class_names)

        global_counts = {}
        for content, _ in training_data:
            for word in tokenize(content):
                global_counts[word] = global_counts.get(word, 0) + 1
        
        filtered_words = sorted([w for w, c in global_counts.items() if c >= self.min_freq])
        self.vocab = {word: i for i, word in enumerate(filtered_words)}
        vocab_size = len(self.vocab)

        class_counts = np.zeros((num_classes, vocab_size), dtype=np.float32)
        class_doc_counts = np.zeros(num_classes, dtype=np.float32)
        
        for content, class_name in training_data:
            c_idx = self.class_map[class_name]
            class_doc_counts[c_idx] += 1
            for word in tokenize(content):
                if word in self.vocab:
                    class_counts[c_idx, self.vocab[word]] += 1

        self.log_priors = np.log(class_doc_counts / len(training_data))

        numerator = class_counts + 1
        denominator = class_counts.sum(axis=1, keepdims=True) + vocab_size
        self.log_likelihoods = np.log(numerator / denominator)

    def predict(self, documents):
        vocab_size = len(self.vocab)
        predictions = []
        
        for content in documents:
            doc_vec = np.zeros(vocab_size)
            for word in tokenize(content):
                if word in self.vocab:
                    doc_vec[self.vocab[word]] += 1
            
            scores = np.dot(doc_vec, self.log_likelihoods.T) + self.log_priors
            pred_idx = np.argmax(scores)
            
            predictions.append(self.index_map[pred_idx])
            
        return predictions

if __name__ == "__main__":
    start_time = time.time()

    train_data, test_data, classes = load_data(DATASET_PATH)

    model = MultinomialNB()
    model.train(train_data, classes)

    test_contents = [t[0] for t in test_data]
    test_labels = [t[1] for t in test_data]
    
    predictions = model.predict(test_contents)
    
    correct = sum(1 for p, a in zip(predictions, test_labels) if p == a)
    accuracy = (correct / len(test_data)) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time: {time.time() - start_time:.2f}s")