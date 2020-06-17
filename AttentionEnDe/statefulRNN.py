import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer as tokenizer


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


def preprocess(texts):
    x = np.array(tokenizer.texts_to_sequences(texts)) - 1

    return tf.one_hot(x, max_id)


def next_char(text, temperature=1):
    x_new = preprocess([text])
    y_proba = model.predict(x_new)[0, -1:, :]
    rescaled_logit = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logit, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


if __name__ == '__main__':
    shakespeare_url = "https://homl.info/shakespeare"
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()

    # Then Encode char to integer
    # Use Tokenizer class: allow to vectorize text corpus by tuning each text to sequence of integer
    # or into a vector
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts([shakespeare_text])

    tokenizer.texts_to_sequences(["First"])
    tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])

    max_id = len(tokenizer.word_index)  # number of distinct chars
    # dataset_size = tokenizer.document_count  # total number of chars

    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
    dataset_size = encoded.shape[0]
    train_size = dataset_size * 90 // 100
    # Slice for get 90% to dataset
    a = encoded[:train_size]
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

    n_steps = 100
    window_length = n_steps + 1  # target = input shifted 1 character ahead
    # Window method create several windows with length = 101, 1st window contain 0 -> 100
    # second one contain 1-> 101, then flatten all of window
    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

    # Convert to one-hot vectors
    dataset = dataset.map(
        lambda x_batch, y_batch: (
            tf.one_hot(x_batch, depth=max_id),
            y_batch
        )
    )
    dataset = dataset.prefetch(1)

    model = keras.models.Sequential([
        keras.layers.GRU(
            128,
            return_sequences=True,
            stateful=True,
            input_shape=[None, max_id],
            dropout=0.2, recurrent_dropout=0.2
        ),
        keras.layers.GRU(
            128,
            return_sequences=True,
            stateful=True,
            dropout=0.2,
            recurrent_dropout=0.2
        ),
        keras.layers.TimeDistributed(
            keras.layers.Dense(max_id, activation="softmax")
        )
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])

    X_new = preprocess("How are yo")
    y_pred = model.predict(X_new)
    tokenizer.sequences_to_texts(y_pred + 1)[0][-1]

    print(complete_text("t", temperature=0.2))