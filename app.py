import streamlit as st
import core.pipeline as pipeline
import tempfile
import os


def save_uploaded_file_to_temp(upload):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1]) as temp_file:
        temp_file.write(upload.read())
        return temp_file.name

st.title("Welcome to :green[CENSE.io!] :sparkler:")
st.subheader("Here, you can test out the current AI algorithm for our thesis project.")

st.text("Cense is an AI powered algorithm that utilizes YOLO and ResNet models in detection and classification in order to provide real-time content moderation by securely detecting and censoring underage children’s faces on live streaming platforms.")
st.write("The AI you're about to test is just a module only and would be the :red[core component] of a :green[bigger] system, as our goal is to :green[automate] content moderation in livestreaming platforms. With this, the system is set to cater video :green[inputs] and :green[outputs]. This webpage is just a :red[trial] run in order for us to visualize on how would the AI process a single video frame.")
uploaded_file = st.file_uploader(":green[Choose an image to test.]")

if uploaded_file is not None:
    st.write(":green[File uploaded. Please wait!]")
else:
    st.write(":red[No file uploaded.]")

st.header("Uploaded Image")
if uploaded_file is not None:
    imagePath = save_uploaded_file_to_temp(uploaded_file)
    st.image(uploaded_file, caption="Original Image")
    IMG, tt = pipeline.returnAnalysis(imagePath)
    print(f"Extraction Time: {round(tt, 3)} ms")
    st.header("Result Image")
    st.image(IMG, caption='Result    Image')
else:
    st.image("static/placeholder.jpg", caption='No image selected.')
    st.header("Result image")
    st.image("static/placeholder.jpg", caption='No result image available.')

