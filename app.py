import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time
st.set_page_config(page_title="Male vs Female Eye Classifier",page_icon="👁️",layout="wide")
@st.cache_resource
def load_model(): return tf.keras.models.load_model("model.keras")
model=load_model()
st.title("👁️ Male vs Female Eye Classifier")
up=st.file_uploader("Upload",type=["jpg","jpeg","png"])
if up:
 img=Image.open(up).convert("RGB"); st.image(img,use_container_width=True)
 if st.button("Analyze"):
  arr=np.expand_dims(np.array(img.resize((299,299)))/255.0,0)
  p=float(model.predict(arr,verbose=0)[0][0])
  male=p*100; female=(1-p)*100
  label="👨 Male Eye" if p>=0.5 else "👩 Female Eye"
  conf=max(p,1-p)*100
  st.success(f"{label} ({conf:.2f}%)")
  fig=go.Figure(go.Bar(x=["Male","Female"],y=[male,female]))
  fig.update_layout(template="plotly_dark")
  st.plotly_chart(fig,use_container_width=True)
