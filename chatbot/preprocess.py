import pandas as pd
import numpy as np
import nltk
import json
import re
import tensorflow as tf 
from nltk.tokenize import word_tokenize




nltk.download('punkt')

model_path = 'C:/Users/admin/Desktop/SIH buzzy bots/Testings-Only/my_model_shitty.h5'


def load_data(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=['line_id', 'character_id', 'movie_id', 'character_name', 'dialogue'], on_bad_lines='skip')
        print(data.head()) 
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  
    
    inputs = data['dialogue'][:-1].tolist()  
    responses = data['dialogue'][1:].tolist()  
    
    return pd.DataFrame({'input': inputs, 'response': responses})

def clean_text(text):
    if isinstance(text, str):  
        text = text.lower()  
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    else:
        text = ''  
    return text


def preprocess_data(file_path):
    
    data = load_data(file_path)

    
    data.dropna(subset=['input', 'response'], inplace=True)
    
    inputs = data['input'].tolist()
    responses = data['response'].tolist()

    
    cleaned_inputs = [clean_text(text) for text in inputs]
    cleaned_responses = [clean_text(text) for text in responses]

    
    
    tokenized_inputs = [word_tokenize(text) for text in cleaned_inputs]
    tokenized_responses = [word_tokenize(text) for text in cleaned_responses]

    
    vocab = set(word for text in tokenized_inputs + tokenized_responses for word in text)
    vocab_size = len(vocab)

    
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    with open('word_to_index.json', 'w') as f:
        json.dump(word_to_index, f)

    with open('index_to_word.json', 'w') as f:
        json.dump(index_to_word, f)

    
    encoder_input_data = [[word_to_index[word] for word in text] for text in tokenized_inputs]
    decoder_input_data = [[word_to_index[word] for word in text] for text in tokenized_responses]

    
    max_length = max(max(len(seq) for seq in encoder_input_data), max(len(seq) for seq in decoder_input_data))
    encoder_input_data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in encoder_input_data])
    decoder_input_data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in decoder_input_data])

    return encoder_input_data, decoder_input_data, vocab_size, word_to_index, index_to_word


seq2seq_model = tf.keras.models.load_model(model_path)



if __name__ == "__main__":
    file_path = r'C:/Users/admin/Desktop/SIH buzzy bots/movie_lines.tsv' 
    encoder_input_data, decoder_input_data, vocab_size, word_to_index, index_to_word = preprocess_data(file_path)
    
    
    print("Preprocessing complete.")
    print(f"Vocabulary size: {vocab_size}")
    print("Model loaded from:", model_path)


