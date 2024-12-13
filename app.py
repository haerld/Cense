import streamlit as st

st.title("Welcome to :green[CENSE!] :sunglasses:")
st.subheader("Here you can test out the current model for our project Cense.")

st.text("Cense is an AI-powered algorithm that uses CNN and LSTM models to provide real-time content"
        "moderation by securely detecting and censoring underage children’s faces on live streaming platforms.")

uploaded_file = st.file_uploader(":green[Choose an image.]")

st.header("Uploaded Image")
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image')
else:
    st.image("static/placeholder.jpg", caption='No image selected.')

st.header("Result image")

st.image("static/placeholder.jpg", caption='No result image available.')