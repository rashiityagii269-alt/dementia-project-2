import streamlit as st
import pickle

st.title("AI Dementia Detection Tool")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

user_input = st.text_area("Enter your sentence:")

if st.button("Check"):
    if user_input:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk")
    else:
        st.warning("Please enter some text")