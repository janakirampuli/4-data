import os
import random
import fasttext

def combine_and_shuffle(pos_file: str, neg_file: str, output_file: str):

    lines = []
    
    if os.path.exists(pos_file):
        with open(pos_file, 'r', encoding='utf-8') as f:
            lines.extend(f.readlines())
    else:
        print(f"file not found {pos_file}")

    if os.path.exists(neg_file):
        with open(neg_file, 'r', encoding='utf-8') as f:
            lines.extend(f.readlines())
    else:
        print(f"file not found {neg_file}")

    if not lines:
        raise ValueError("no data")

    print(f"total samples gathered: {len(lines)}")
    print("shuffling data...")
    random.seed(42) 
    random.shuffle(lines)

    print(f"writing data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
    return len(lines)

def train_model(train_file: str, model_out_path: str):

    model = fasttext.train_supervised(
        input=train_file,
        lr=0.1,                # Learning rate
        epoch=5,               # Number of epochs
        wordNgrams=2,          # Use bigrams
        dim=50,                # Size of word vectors
        bucket=200000,         # Number of buckets for vocabulary
        loss='hs'
    )
    
    print(f"saving trained model to {model_out_path}...")
    model.save_model(model_out_path)
    
    result = model.test(train_file)
    print("Sanity Check (Performance on Training Data):")
    print(f"Total Samples Tested: {result[0]}")
    print(f"Precision: {result[1]:.4f}")
    print(f"Recall: {result[2]:.4f}")
    
    return model

if __name__ == "__main__":
    pos_data_path = "data/train_positive.txt"
    neg_data_path = "data/train_negative.txt"
    combined_data_path = "data/train_combined.txt"
    
    model_output_path = "quality_classifier.bin"

    try:
        combine_and_shuffle(pos_data_path, neg_data_path, combined_data_path)
        train_model(combined_data_path, model_output_path)
    except Exception as e:
        print(f"error: {e}")