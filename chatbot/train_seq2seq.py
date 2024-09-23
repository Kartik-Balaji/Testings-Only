import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from preprocess import preprocess_data  # Import the preprocess function

# Path to your dataset
file_path = 'C:/Users/asus/Desktop/Github/Testings-Only/movie_lines.tsv'

# Preprocess the data
encoder_input_data, decoder_input_data, vocab_size = preprocess_data(file_path)

# Define hyperparameters
latent_dim = 128  # or 64
max_sequence_length = encoder_input_data.shape[1]  # The max length of the padded sequences

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the Seq2Seq model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Print the model summary
model.summary()

# You also need to prepare decoder target data by shifting decoder input sequences by one timestep.
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# Train the model (you might want to tune the batch size and number of epochs)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=100, validation_split=0.2)
model.save("MyShit.h5")