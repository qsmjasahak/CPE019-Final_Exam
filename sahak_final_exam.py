import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('/content/drive/MyDrive/CPE019-CPE32S4/drowsiness dataset/drowsiness_classification.h5')
  return model
model=load_model()
st.write("""
# Drowsiness Classification"""
)
file=st.file_uploader("Upload a photo from your computer.",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(60,60)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image related to drowsiness.")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names = ['yawn', 'no_yawn', 'Closed', 'Open']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
