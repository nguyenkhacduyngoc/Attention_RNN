import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np

sequence_lengths = 50
vocab_size = 10000
embed_size = 512

# Init input for head of encoder and decoder
encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)

# Change from word to embedded number, each word represented by its ID
embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

# Training decoder:
# Use LSTM with 512 output and choose to return the last state in addition to the output
# (it means providing access to the hidden state (state_h) and cell state (state_c))
encoder = keras.layers.Bidirectional(keras.layers.LSTM(512, return_state=True))
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

# Training encoder
sampler = tfa.seq2seq.sampler.TrainingSampler()
decoder_cell = keras.layers.LSTMCell(512)
output_layer = keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                 output_layer=output_layer)

final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state,
    sequence_length=sequence_lengths)

Y_proba = tf.nn.softmax(final_outputs.rnn_output)

model = keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths], outputs=[Y_proba])
