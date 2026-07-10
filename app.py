import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Male vs Female Eye Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
}

.hero{
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    padding:35px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 8px 25px rgba(0,0,0,0.3);
}

.hero h1{
    font-size:52px;
    margin-bottom:10px;
}

.hero p{
    font-size:20px;
}

.card{
    background:rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius:18px;
    padding:25px;
    box-shadow:0 8px 25px rgba(0,0,0,.25);
}

.result-card{
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    color:white;
    border-radius:20px;
    padding:30px;
    text-align:center;
    box-shadow:0px 8px 25px rgba(0,0,0,.3);
}

.metric{
    font-size:45px;
    font-weight:bold;
}

footer{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("👁️ About")

    st.info("""
This application predicts whether an eye image belongs to a **Male** or **Female** using a Convolutional Neural Network.

### Model
- CNN
- TensorFlow/Keras

### Input
- JPG
- JPEG
- PNG

### Output
- Predicted Gender
- Confidence Score
""")

    st.success("Developed using Streamlit")

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class='hero'>
<h1>👁️ Male vs Female Eye Classifier</h1>
<p>Deep Learning • CNN • TensorFlow • Streamlit</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Layout
# -----------------------------
left,right = st.columns([1,1])

uploaded = None

with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📂 Upload Eye Image")

    uploaded = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🖼️ Image Preview")

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(image,use_container_width=True)

    else:

        st.info("Upload an eye image to preview.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if uploaded:

    image = Image.open(uploaded).convert("RGB")

    img = image.resize((299,299))      # Change if required

    img = np.array(img)/255.0

    img = np.expand_dims(img,axis=0)

    st.write("")

    if st.button("🚀 Analyze Image",use_container_width=True):

        with st.spinner("Analyzing..."):

            pred = model.predict(img)

        probability = float(pred[0][0])

        if probability >= 0.5:

            label = "👨 Male Eye"

            confidence = probability

        else:

            label = "👩 Female Eye"

            confidence = 1-probability

        st.write("")

        st.markdown(f"""
        <div class='result-card'>
            <h2>Prediction Result</h2>
            <div class='metric'>{label}</div>
            <h3>Confidence : {confidence*100:.2f}%</h3>
        </div>
        """,unsafe_allow_html=True)

        st.write("")

        st.subheader("📊 Confidence")

        st.progress(float(confidence))

        male_prob = probability
        female_prob = 1-probability

        st.subheader("📈 Prediction Probability")

        st.bar_chart({
            "Probability":[male_prob,female_prob]
        })

        c1,c2 = st.columns(2)

        c1.metric("👨 Male",f"{male_prob*100:.2f}%")
        c2.metric("👩 Female",f"{female_prob*100:.2f}%")

st.write("")
st.write("---")

st.markdown(
"""
<center>
<h4>✨ Built with TensorFlow • Keras • Streamlit</h4>
Made by <b>Anushka Singh</b> 💙
</center>
""",
unsafe_allow_html=True
)
