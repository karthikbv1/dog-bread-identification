import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import json
import os
import requests
from streamlit_lottie import st_lottie
import io
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------
# CONFIGURATION
# ----------------------------------
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="wide"
)

if 'page' not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_prediction_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("imagenet_model.keras")
    return model

@st.cache_data
def load_data():
    labels = []
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f]

    breed_info = {}
    if os.path.exists("data/breed_info.json"):
        with open("data/breed_info.json", "r", encoding="utf-8") as f:
            breed_info = json.load(f)

    return labels, breed_info

model = load_prediction_model()
labels, BREED_INFO = load_data()

# ----------------------------------
# UI STYLE
# ----------------------------------
st.markdown("""
<style>
.stApp {background:#f4f6f8;}
.title {font-size:40px;font-weight:bold;color:#333;}
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 10px #ddd;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# NAVBAR
# ----------------------------------
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    if st.button("🏠 Home"): set_page("Home")
with c2:
    if st.button("📖 About"): set_page("About")
with c3:
    if st.button("🔍 Predict"): set_page("Predict")
with c4:
    if st.button("📊 Model"): set_page("Model")
with c5:
    if st.button("🐾 Care"): set_page("Care")

st.markdown("---")

# ----------------------------------
# IMAGE PREPROCESS
# ----------------------------------
def preprocess_image(img):
    img = img.resize((331,331))
    img = np.array(img)
    img = np.expand_dims(img,0)
    img = (img/127.5)-1
    return img

# ----------------------------------
# PAGES
# ----------------------------------

def show_home():
    st.markdown("<div class='title'>🐶 Dog Breed Classification System</div>", unsafe_allow_html=True)
    st.write("""
    A Deep Learning based web application to identify dog breeds from images.  
    Built using **TensorFlow + Streamlit**.
    """)

    st.image("https://i.imgur.com/q3ZJ6FJ.png", width=600)

    st.markdown("""
    ### ❤️ Why this project?
    - Helps dog lovers identify breeds  
    - Useful for vets & shelters  
    - Educational for ML students  
    """)


def show_about():
    st.title("📖 About This Project")

    st.markdown("""
    <div class="card">
    <h3>👨‍💻 Developed by</h3>
    Karthik B.V (MCA - AI & ML)

    <h3>📌 Project Description</h3>
    This project uses a **Deep Learning CNN model** trained on dog breed images  
    to classify **200+ dog breeds** accurately.

    <h3>🛠 Technologies Used</h3>
    - Python  
    - TensorFlow / Keras  
    - Streamlit  
    - NumPy, Pandas, Matplotlib  

    </div>
    """, unsafe_allow_html=True)


def show_model():
    st.title("📊 Model Information")

    st.markdown("""
    <div class="card">
    <h3>🔍 Model Type</h3>
    Transfer Learning CNN (ImageNet pretrained)

    <h3>📈 Accuracy</h3>
    Training Accuracy: **93%**  
    Validation Accuracy: **94%**

    <h3>🧠 What model learned?</h3>
    - Facial patterns  
    - Fur texture  
    - Ear shapes  
    - Body structure  

    <h3>🎯 Use Cases</h3>
    - Pet adoption platforms  
    - Veterinary clinics  
    - Animal shelters  
    - Dog training centers  
    - Mobile apps  

    </div>
    """, unsafe_allow_html=True)


def show_predict():
    st.title("🔍 Predict Dog Breed")

    file = st.file_uploader("Upload dog image",type=["jpg","png","jpeg"])

    if file:
        col1,col2 = st.columns(2)
        with col1:
            img = Image.open(file)
            st.image(img,caption="Uploaded Image",use_column_width=True)

        with col2:
            if st.button("🚀 Run Prediction"):
                arr = preprocess_image(img)
                preds = model.predict(arr)

                idx = np.argmax(preds)
                conf = preds[0][idx]*100
                breed = labels[idx]

                st.success(f"🐕 Breed : {breed}")
                st.info(f"🎯 Confidence : {conf:.2f}%")

                # TOP 5
                top5 = np.argsort(preds[0])[-5:][::-1]
                names = [labels[i] for i in top5]
                values = preds[0][top5]*100

                df = pd.DataFrame({"Confidence":values},index=names)
                st.subheader("Top 5 Predictions")
                st.bar_chart(df)


def show_care():
    st.title("🐾 Dog Care Guide")

    st.markdown("""
    <div class="card">
    <h3>🍖 Feeding</h3>
    High protein diet, fresh water always.

    <h3>🏃 Exercise</h3>
    Daily walks, play time.

    <h3>🛁 Grooming</h3>
    Regular brushing & bathing.

    <h3>💉 Health</h3>
    Vaccination, vet checkups.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------
# ROUTER
# ----------------------------------
if st.session_state.page=="Home":
    show_home()
elif st.session_state.page=="About":
    show_about()
elif st.session_state.page=="Predict":
    show_predict()
elif st.session_state.page=="Model":
    show_model()
elif st.session_state.page=="Care":
    show_care()
