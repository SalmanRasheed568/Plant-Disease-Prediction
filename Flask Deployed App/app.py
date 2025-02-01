import os
import streamlit as st
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load disease and supplement info
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load CNN model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_Entire.pt"))
model.eval()

# Define prediction function
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Create Streamlit app
st.title("Plant Disease Detector")

# Create file uploader
uploaded_file = st.file_uploader("Upload an image of a plant leaf")

# Create prediction button
if st.button("Make a prediction"):
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Make prediction
        pred = prediction(temp_file_path)

        # Get disease and supplement info
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        # Display prediction results
        st.write("Disease detected: ", title)
        st.write("Description: ", description)
        st.write("Prevention steps: ", prevent)
        st.image(image_url)
        st.write("Supplement name: ", supplement_name)
        st.image(supplement_image_url)
        st.write("Buy link: ", supplement_buy_link)

# Create market page
st.write("Market")
supplement_image = list(supplement_info['supplement image'])
supplement_name = list(supplement_info['supplement name'])
disease = list(disease_info['disease_name'])
buy = list(supplement_info['buy link'])

# Display market info
for i in range(len(supplement_image)):
    st.write("Supplement name: ", supplement_name[i])
    st.image(supplement_image[i])
    st.write("Disease: ", disease[i])
    st.write("Buy link: ", buy[i])
