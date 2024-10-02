import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

seq2seq_model = load_model('my_model_shitty.h5')

with open('word_to_index.json', 'r') as f:
    word_to_index = json.load(f)

with open('index_to_word.json', 'r') as f:
    index_to_word = json.load(f)

def generate_response(input_text, model, word_to_index, index_to_word, max_length):
    
    input_seq = [word_to_index.get(word, word_to_index.get('<UNK>', -1)) for word in input_text.split()]

    
    input_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')

    
    decoded_seq = model.predict(input_seq)

    
    predicted_indices = np.argmax(decoded_seq, axis=-1)

    
    response = ' '.join([index_to_word.get(int(idx), '<UNK>') for idx in predicted_indices[0] if int(idx) != 0])
    
    if not response.strip(): 
        response = "Dont ask me anything other than booking a ticket my nigga."
    
    return response



while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_response(user_input, seq2seq_model, word_to_index, index_to_word, max_length=20)
    print(f"Bot: {response}")
