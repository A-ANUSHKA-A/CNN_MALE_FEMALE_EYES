import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Male vs Female Eye Classifier",
    page_icon="👁️",
    layout="wide"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>

.main{
background:linear-gradient(135deg,#141E30,#243B55);
}

.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:white;
}

.subtitle{
text-align:center;
font-size:18px;
color:#d6d6d6;
margin-bottom:25px;
}

.result{
padding:20px;
border-radius:15px;
background:#ffffff20;
text-align:center;
font-size:28px;
font-weight:bold;
color:white;
}

.stButton>button{
width:100%;
background:#00B4DB;
background:linear-gradient(to right,#0083B0,#00B4DB);
color:white;
font-size:18px;
border-radius:12px;
height:55px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -----------------------
# Header
# -----------------------
st.markdown("<div class='title'>👁️ Male vs Female Eye Classifier</div>",
unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Deep Learning | CNN | Streamlit</div>",
unsafe_allow_html=True)

left,right=st.columns([1,1])

with left:

    uploaded=st.file_uploader(
        "Upload Eye Image",
        type=["jpg","jpeg","png"]
    )

with right:

    if uploaded is not None:

        image=Image.open(uploaded).convert("RGB")

        st.image(image,use_container_width=True)

        img=image.resize((299,299))      # Change if needed

        img=np.array(img)/255.0

        img=np.expand_dims(img,axis=0)

        if st.button("🔍 Predict"):

            with st.spinner("Analyzing..."):

                pred=model.predict(img)

                probability=float(pred[0][0])

                if probability>0.5:
                    label="👨 Male Eye"
                    confidence=probability*100
                else:
                    label="👩 Female Eye"
                    confidence=(1-probability)*100

                st.markdown(
                f"""
                <div class='result'>
                {label}<br><br>
                Confidence : {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True)

                st.progress(confidence/100)

                st.write("### Prediction Scores")

                st.write({
                    "Male":round(probability,3),
                    "Female":round(1-probability,3)
                })

st.markdown("---")
st.caption("Made ANUSHKA ⭐using TensorFlow and Streamlit")
