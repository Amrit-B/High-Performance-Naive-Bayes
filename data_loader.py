
import os
import re

def load_data(dataset_dir,train_test_split=500):

    training_data = []
    testing_data = []
    
    class_names = sorted([d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))]) #sorted for Consistent class indices 
    
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        filenames = sorted([f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f))]) #sorted for better train/test split

        for idx, fname in enumerate(filenames):
            filepath = os.path.join(class_path, fname)
            with open(filepath, 'r', encoding='latin-1') as f:  #latin-1 is better coz  it preserves original characters 
                content = f.read()
            
            if idx < train_test_split:
                training_data.append((content, class_name))
            else:
                testing_data.append((content, class_name))

    return training_data, testing_data, class_names


def tokenize(text):

    return re.findall(r'[a-zA-Z]+', text.lower()) # Try NLTK next might handle contractions better?
