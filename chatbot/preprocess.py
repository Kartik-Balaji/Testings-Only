import pandas as pd
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources (only need to run once)
nltk.download('punkt')

# Updated load_data() function
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=['line_id', 'character_id', 'movie_id', 'character_name', 'dialogue'], on_bad_lines='skip')
        print(data.head())  # Add this line to check the data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    
    inputs = data['dialogue'][:-1].tolist()  # All but the last line
    responses = data['dialogue'][1:].tolist()  # All but the first line (shifted)
    
    return pd.DataFrame({'input': inputs, 'response': responses})

def clean_text(text):
    if isinstance(text, str):  # Check if the text is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation and special characters
    else:
        text = ''  # Handle missing or non-string values (e.g., NaN)
    return text
from nltk.tokenize import word_tokenize

def preprocess_data(file_path):
    # Load the data
    data = load_data(file_path)

    # Drop rows where 'dialogue' is missing
    data.dropna(subset=['input', 'response'], inplace=True)
    
    inputs = data['input'].tolist()
    responses = data['response'].tolist()

    # Clean and tokenize the text
    cleaned_inputs = [clean_text(text) for text in inputs]
    cleaned_responses = [clean_text(text) for text in responses]

    # Tokenize using the NLTK word_tokenize function, explicitly specifying the language as 'english'
    tokenized_inputs = [word_tokenize(text, language='english') for text in cleaned_inputs]
    tokenized_responses = [word_tokenize(text, language='english') for text in cleaned_responses]

    # Create a vocabulary
    vocab = set(word for text in tokenized_inputs + tokenized_responses for word in text)
    vocab_size = len(vocab)

    # Create a mapping from words to integers
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    # Convert tokens to integer sequences
    encoder_input_data = [[word_to_index[word] for word in text] for text in tokenized_inputs]
    decoder_input_data = [[word_to_index[word] for word in text] for text in tokenized_responses]

    # Pad sequences (using NumPy)
    max_length = max(max(len(seq) for seq in encoder_input_data), max(len(seq) for seq in decoder_input_data))
    encoder_input_data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in encoder_input_data])
    decoder_input_data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in decoder_input_data])

    return encoder_input_data, decoder_input_data, vocab_size


# Example usage
if __name__ == "__main__":
    file_path = 'C:/Users/admin/Desktop/movie_lines.tsv'  # Corrected path to your dataset
    encoder_input_data, decoder_input_data, vocab_size = preprocess_data(file_path)
    print("Preprocessing complete.")
    print(f"Vocabulary size: {vocab_size}")
