from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dropout, Dense, Embedding
from tensorflow.keras.models import Sequential


# Build SimpleRNN Model
def model_simple_rnn(vocab_size, embedding_size, input_length, n_units, dropout):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_length))
    model.add(SimpleRNN(n_units))
    model.add(Dropout(dropout))
    # model.add(Dense(800, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    return model


# Build LSTM Model
def model_lstm(vocab_size, embedding_size, input_length, n_units, dropout):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_length))
    model.add(LSTM(n_units))
    model.add(Dropout(dropout))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    return model


# Build GRU Model
def model_gru(vocab_size, embedding_size, input_length, n_units, dropout):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=input_length))
    model.add(GRU(n_units, return_sequences=True))
    model.add(GRU(n_units))
    model.add(Dropout(dropout))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    return model
