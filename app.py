import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tf_keras

def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

ps = PorterStemmer()
nltk.download('stopwords')

model = tf_keras.models.load_model("model.h5", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("Fake News Detection")
user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    processed_text = preprocess(user_input)
    seq = tokenizer.texts_to_sequences([processed_text])
    pad = pad_sequences(seq, maxlen=20)
    prediction = model.predict(pad)[0][0]
    if prediction > 0.5:
        st.error("Fake News ❌")
    else:
        st.success("Real News ✅")
