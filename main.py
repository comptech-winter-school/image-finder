import streamlit as st
from PIL import Image
import random
import pandas as pd
import abc
import torch
import ruclip
import clip
import numpy as np
from numbers import Number
from typing import List
import os
import glob
import math
from pathlib import Path
import regex as re
import requests
from io import BytesIO
import time
from googletrans import Translator

from searchmodel import SearchModel
from embedder import EmbedderRuCLIP, EmbedderCLIP
from dummyindexer import DummyIndexer
from hnsw_indexer import HnmsIndexer

st.set_page_config(page_title="Image Finder",
                   page_icon='⚙',
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items=None)

with st.expander("About"):
    st.text("""
        Image Finder project.
        
        FAQ:
        1. Select preferred indexer
        2. Select text query or image method for processing
        3. Select output image count
        4. If you want to filter output results, you can use threshold slider
        5. The images will be print with sorting of cosine distance
        
        Models: ruCLIP / CLIP
        
        Indexers:
        1. Tour - curator's images from travel
        2. Video Trailer - Video trailer a first film about Harry Potter
        3. Professional photos - almost two million different images from the photographer from the site Unsplash
        4. Parking - a video where there is a fairly dense traffic of vehicles
        
        Team:
        Developers: Anna Glushkova, Kirill Keller, Alexandr Minin, Maxim Mashtakov,
        Vladislav Kuznetsov, Dmitry Moskalev, Vasiliy Dronov
        Team Lead: Dmitry Moskalev
        Mentors: Amir Uteuov, Vladimir Kilyazov      
    """)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_unsplash_indexer(pth):
    hnsw_indexer = HnmsIndexer()
    hnsw_indexer.load(pth)
    return hnsw_indexer

hnsw_indexer = load_unsplash_indexer('/mnt/storage/unsplash_nms.index')

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
	#CLIP
	clip_model = SearchModel(EmbedderCLIP(device='cpu'), DummyIndexer())
	#ruCLIP
	ruclip_model = SearchModel(EmbedderRuCLIP(device='cpu'), DummyIndexer())
	return clip_model, ruclip_model

clip_model, ruclip_model = load_model()

#Functions
def function_images(input_format):
    data = []
    
    input_format = dict(sorted(input_format.items(), key=lambda item: item[1], reverse=True))

    col1, col2 = st.columns([1, 1])
    count = 0
    for i in input_format.items():
        if i[1] >= threshold / 100:
            count += 1
            df = pd.DataFrame()
            data.append([i[0].split('/')[len(i[0].split('/'))-1], i[1]])
            df = df.append(data)
            df = df.rename(columns={0: "Image", 1: "Cosine distance"})

    if count == 0:
        st.info(f"No images for {threshold} or above cosine distance, %")
    else:
        for _, i in enumerate(input_format.items(), 1):
            with col1:
                try:
                    if i[1] >= threshold / 100:
                        image = Image.open(i[0]) if indexer_name != 'unsplash' else get_image_url(i[0])
                        caption = f"{i[0].split('/')[len(i[0].split('/'))-1]}, {i[1]}" if indexer_name != 'unsplash' else f"{i[0]}, {i[1]}"
                        st.image(image, caption=caption)
                except:
                    pass
            with col2:
                if _ == 1:
                    st.write("Output cosine distance")
                    st.dataframe(df.style.text_gradient(axis=0, cmap='Spectral'))

def get_image_url(photo_id):
  photo_image_url = f"https://unsplash.com/photos/{photo_id}/download?w=320"
  response = requests.get(photo_image_url)
  image = Image.open(BytesIO(response.content))
  return image

st.image(Image.open('assets/logo.png'))

dict_indexer = {'Tour':'trip', 'Video Trailer':'trailer', 'Professional photos':'unsplash', 'Parking': 'traffic'}

#st.title('Image Finder project')
indexer = st.selectbox(
    "Select usecase",
    ("Tour", "Video Trailer", "Professional photos", "Parking")
)
st.caption(f"Current usecase: {indexer}")
option = st.selectbox(
    'What would you like to do?',
     ('Text query', 'Image')
)
#Text query
if option == 'Text query':
    text = st.text_input('Input text query', value = 'Sunset')
else:
#Image
    file = st.file_uploader("Choose image", type=['jpeg', 'jpg', 'png'], accept_multiple_files=False)

values = st.slider('Select a range of sample images', 1, 10, 5)
threshold = st.slider('Select a threshold for output images, %', 1, 100, 10)

if dict_indexer.get(indexer) == 'unsplash':
    use_hnsw = st.checkbox('❗ Use fast indexer 😎', value = True)

translator = Translator()

#Processing
if st.button('Start processing'):
    indexer_name = dict_indexer.get(indexer)
    if option == 'Text query' and text:
        st.write(f"Output images for text query: {text}")
        
        model_prefix = 'RuCLIP'
        general_model = ruclip_model
        
        if re.findall(r'[а-яА-Я0-9]', text):
            model_prefix = 'RuCLIP'
            general_model = ruclip_model
        elif re.findall(r'[a-zA-Z0-9]', text):
            model_prefix = 'CLIP'
            general_model = clip_model
        else:
            st.info(f"Error in query: {text}")

        if indexer_name == 'unsplash' and model_prefix == 'RuCLIP':
            text = translator.translate(text, src = 'ru', dest='en').text
            general_model = clip_model
        
        model_prefix = model_prefix if indexer_name != 'unsplash' else 'others'
        general_model.load_imgs(f"/home/comptech/indexes/{indexer_name}/images", model_prefix)
        general_model.indexer.load(str(general_model.features_path) + '/features.npy')
        start_time = time.time() * 1000
        query = general_model.embedder.encode_text(text)
        if indexer_name == 'unsplash' and use_hnsw:
            distances_hnsw, indexes_hnsw = hnsw_indexer.find(query, values)
            distances_hnsw = np.array(distances_hnsw)
            l_query = []
            for l_item in indexes_hnsw:
                l_query.append(general_model.imgs_path[l_item])
            indexes_hnsw = np.array(l_query)
            input_data = (distances_hnsw, indexes_hnsw)
        else:
            try:
                input_data = general_model.get_k_imgs(query, values)
            except:
                input_data = general_model.get_k_imgs(query, values)
        end_time = time.time()*1000
        input_format = {}
        st.write("Search time in ms: " + str(round((end_time-start_time),2)))
        
        for i,j in zip(input_data[0], input_data[1]):
            input_format.update({str(j):i})
        
        function_images(input_format)
        
    elif option == 'Image' and file is not None:
        model_prefix = 'CLIP' if indexer_name != 'unsplash' else 'others'
        clip_model.load_imgs(f"/home/comptech/indexes/{indexer_name}/images", model_prefix)
        clip_model.indexer.load(str(clip_model.features_path) + '/features.npy')
        image = Image.open(file)
        start_time = time.time() * 1000
        query = clip_model.embedder.encode_imgs([image])
        st.image(image, caption=file.name)
        st.write(f"Output images for current input image: {file.name}")
        if indexer_name == 'unsplash' and use_hnsw:
            distances_hnsw, indexes_hnsw = hnsw_indexer.find(query, values)
            distances_hnsw = np.array(distances_hnsw)
            l_query = []
            for l_item in indexes_hnsw:
                l_query.append(clip_model.imgs_path[l_item])
            indexes_hnsw = np.array(l_query)
            input_data = (distances_hnsw, indexes_hnsw)
        else:
            try:
                input_data = clip_model.get_k_imgs(query, values)
            except:
                input_data = clip_model.get_k_imgs(query, values)
        end_time = time.time() * 1000
        st.write("Search time in ms: " + str(round((end_time-start_time), 2)))
        input_format = {}
        
        for i,j in zip(input_data[0], input_data[1]):
            input_format.update({str(j):i})
        
        function_images(input_format)
        
    else:
        st.info("Error")