import streamlit as st
from PIL import Image
import random
import pandas as pd


#Functions
def function_images():
    cosine_distance, data = [], []

    for i in range(values):
        cosine_distance.append(random.random())

    input_format = {f'({i+1}).jpg': j for i, j in zip(range(values), cosine_distance)}
    input_format = dict(sorted(input_format.items(), key=lambda item: item[1], reverse=True))

    col1, col2 = st.columns([1, 1])

    for i, j in zip(input_format.keys(), input_format.values()):
        data.append([i, j])
        df = pd.DataFrame()
        df = df.append(data)
        df = df.rename(columns={0: "Image", 1: "Cosine distance"})

    for _, i in enumerate(input_format.items(), 1):
        with col1:
            image = Image.open(i[0])
            st.image(image, caption=f'{i[0]}, {i[1]}')
        with col2:
            if _ == 1:
                st.write("Output cosine distance")
                st.dataframe(df.style.text_gradient(axis=0, cmap='Spectral'))


st.image(Image.open('logo.png'))
indexer = st.sidebar.selectbox(
    "Select indexer",
    ("Variant1", "Variant2", "Variant3", "Variant4", "Variant5")
)
st.title('Project')
st.caption(f'Current indexer: {indexer}')
option = st.selectbox(
    'What would you like to do?',
     ('Text query', 'Image')
)

#Text query
if option == 'Text query':
    text = st.text_input('Input text query', '')
else:
#Image
    file = st.file_uploader("Choose image", type=['jpeg', 'jpg', 'png'], accept_multiple_files=False)
values = st.slider('Select a range of sample images', 1, 10, 5)

#Processing
if st.button('Process'):
    if option == 'Text query' and text:
        st.write(f'Output images for text query: {text}')
        function_images()
    elif option == 'Image' and file is not None:
        image = Image.open(file.name)
        st.image(image, caption=file.name)
        st.write(f'Output images for current input image: {file.name}')
        function_images()
    else:
        st.write("Error")
