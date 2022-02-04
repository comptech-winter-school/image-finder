"""
Created on 2022 Jan 30 10:14 
@author: keller
"""
import os
import glob
import numpy as np
import pandas as pd
import math
from PIL import Image
from typing import List
from pathlib import Path


class SearchModel():
    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer
        self.images_dir = None
        self.imgs_path = None
        self.features_path = None

    def load_imgs(self, path: str, prefix: str):
        """
        Returns a list of names images in a given path
        :param path:
        :return:
        """
        self.images_dir = path
        photos_path = Path(self.images_dir)
        general_features_dir = str(photos_path.parents[0]) + '/features'
        features_dir = general_features_dir + '/' + prefix
        self.features_path = Path(features_dir)
        self.imgs_path = list(photos_path.glob("*.*"))
        
        if not os.path.exists(general_features_dir):
          os.mkdir(general_features_dir)
        
        if not os.path.exists(features_dir):
          os.mkdir(features_dir)
        
        if len(os.listdir(features_dir)) >= 2:
          self.imgs_path = list(pd.read_csv(f"{self.features_path}/photo_ids.csv")['photo_id'])

    def load_img_urls(self):
        """
        In case we want to load imgs from a list of url
        :return:
        """
        pass

    def save_embs(self, batch_size=512) -> None:
        """
        Extracts image embeddings from embedder and adds them to indexer
        :param pil_imgs:
        :return:
        """

        if len(os.listdir(self.features_path)) >= 2:
          os.remove(str(self.features_path) + '/photo_ids.csv')
          os.remove(str(self.features_path) + '/features.npy')
          self.imgs_path = list(Path(self.images_dir).glob("*.*"))
        
        if not len(self.imgs_path) >= 512:
          batch_size = len(self.imgs_path)

        # Compute how many batches are needed
        batches = math.ceil(len(self.imgs_path) / batch_size)

        # Process each batch
        for i in range(batches):
          print(f"Processing batch {i+1}/{batches}")

          batch_ids_path = self.features_path / f"{i:010d}.csv"
          batch_features_path = self.features_path / f"{i:010d}.npy"
    
          # Only do the processing if the batch wasn't processed yet
          if not batch_features_path.exists():
            try:
              # Select the photos for the current batch
              batch_files = self.imgs_path[i*batch_size : min(len(self.imgs_path), (i+1)*batch_size)]
              pil_batch = [Image.open(photo_file) for photo_file in batch_files]

              # Compute the features and save to a numpy file
              batch_features = self.embedder.encode_imgs(pil_batch)
              np.save(batch_features_path, batch_features)

              # Save the photo IDs to a CSV file
              photo_ids = [photo_file for photo_file in batch_files]
              photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
              photo_ids_data.to_csv(batch_ids_path, index=False)
            except:
              # Catch problems with the processing to make the process more robust
              print(f'Problem with batch {i}')

        # Load all numpy files
        features_list = [np.load(features_file) for features_file in sorted(self.features_path.glob("*.npy"))]

        # Concatenate the features and store in a merged file
        features = np.concatenate(features_list)
        np.save(self.features_path / "features.npy", features)

        # Load all the photo IDs
        photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(self.features_path.glob("*.csv"))])
        photo_ids.to_csv(self.features_path / "photo_ids.csv", index=False)
        
        for file in glob.glob('{}/0*.*'.format(self.features_path)):
          os.remove(file)
        
        self.indexer.load(str(self.features_path) + '/features.npy')
    
    def get_k_imgs(self, emb: np.ndarray, k: int):
        """
        Returns k indices of nearest image embeddings and respective distances for a given embedding emb
        :param emb:
        :param k:
        :return:
        """
        distances, indices = self.indexer.find(emb, k)
        return distances, np.array(self.imgs_path)[indices]