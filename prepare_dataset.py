import tensorflow as tf
from tensorflow import keras
import numpy as np

shakespeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

# Then Encode char to integer
# Use Tokenizer class: allow to vectorize text corpus by tuning each text to sequence of integer
# or into a vector
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

max_id =len(tokenizer.word_index) # number of distinct chars
dataset_size = tokenizer.document_count # total number of chars

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text]))
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1 # target = input shifted 1 character ahead
dataset = dataset.window(window_length, shift=1, drop_remainder=True)


print("A")


