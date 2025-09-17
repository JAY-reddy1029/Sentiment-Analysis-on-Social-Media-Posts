import pickle

# Load saved files
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Example test
sample = ["I really hate this product, it's awful 😡"]
sample_tfidf = vectorizer.transform(sample)
print("Prediction:", model.predict(sample_tfidf)[0])


import streamlit as st
import pickle

# Load files
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a post/tweet:")

if st.button("Predict"):
    transformed = vectorizer.transform([user_input])
    prediction = model.predict(transformed)[0]
    st.write("Predicted Sentiment:", prediction)
