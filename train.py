import pandas as pd
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from model import model_simple_rnn, model_lstm, model_gru

df = pd.read_csv('train.csv')

# shuffle data
df = shuffle(df)

X_train = df["sentence"].fillna("fillna").values
y_train = df[["BookRestaurant", "GetWeather", "PlayMusic", "RateBook"]].values
#############################################
print(X_train[0])
print(y_train[0])
import sys
sys.exit()
#############################################
#y_train = df[["greeting", "goodbye", "thanks", "ordering", "payments", "delivery"]].values

# Preprocessing

text = X_train
# text preprocessing
Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(text)
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
#print(Tokenizer_vocab_size, X_train.shape, y_train.shape)

####################################################
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, shuffle=True, random_state=0)

####################################################

####################################################
# samples = int(len(X_train)*0.2)
#
# X_train = X_train[samples:]
# y_train = y_train[samples:]
#
# X_val = X_train[:samples]
# y_val = y_train[:samples]
####################################################
X_train_encoded_words = Tokenizer.texts_to_sequences(X_train)
X_val_encoded_words = Tokenizer.texts_to_sequences(X_val)
X_train_encoded_padded_words = sequence.pad_sequences(X_train_encoded_words, maxlen = 4000)
X_val_encoded_padded_words = sequence.pad_sequences(X_val_encoded_words, maxlen = 4000)

# print(X_train_encoded_padded_words.shape, X_val_encoded_padded_words.shape)
# print(X_val_encoded_padded_words, X_train_encoded_padded_words)

# Build and Train the Model
# model = model_simple_rnn(Tokenizer_vocab_size, 32, 4000, 10, 0.1) # Uncomment this line to train Simple RNN Model 
# model = model_lstm(Tokenizer_vocab_size, 32, 4000, 10, 0.5)       # Uncomment this line to train LSTM Model 
model = model_gru(Tokenizer_vocab_size, 32, 4000, 10, 0.2)          # Uncomment this line to train GRU Model 

model.summary()

# Hyper parameters
batch_size = 256
n_epochs = 5
learning_rate = 0.2

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train_encoded_padded_words, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_val_encoded_padded_words, y_val))

# Save Tokenizer for chat.py, loss and val_loss for visualization.py
with open("token_history", "wb") as f:
    pickle.dump([Tokenizer, history.history['loss'], history.history['val_loss']], f)

# save the model
model.save("model_rnn.h5")


# accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='lower right')
# plt.show()

model.evaluate(X_val_encoded_padded_words, y_val)

