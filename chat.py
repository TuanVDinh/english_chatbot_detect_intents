import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import Tokenizer from train.py
with open("token_history", "rb") as f:
    Tokenizer, _, _ = pickle.load(f)

new_model = load_model("model_rnn.h5")


def get_response(msg):
    sentence = msg
    tokens = Tokenizer.texts_to_sequences([sentence])
    tokens = pad_sequences(tokens, maxlen=4000)
    prediction = new_model.predict(np.array(tokens))
    predicted = np.argmax(prediction)
    classes = ['BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook']
    #classes = ["greeting", "goodbye", "thanks", "ordering", "payments", "delivery"]
    s = f'Your intent is "{classes[predicted]}"'
    if np.amax(prediction) > 0.7:
        return s
    else:
        return "Sorry, I do not know your intent. Please try again!"