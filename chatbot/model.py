import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Define hyperparameters
latent_dim = 256  # Dimensionality of the latent space (i.e., the output of the LSTM)
vocab_size = 61925  # Use the vocab size from your preprocessing step
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

# Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Summarize the model architecture
model.summary()
