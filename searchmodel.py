"""
Created on 2022 Jan 30 10:14 

@author: keller
"""
import glob

import numpy as np
from PIL import Image
from typing import List

from dummyindexer import DummyIndexer
from embedder import EmbedderRuCLIP


class SearchModel():
    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer
        self.indexed_imgs_path = [] # array with indexed embeddings
        self.imgs_path = None       # array for temp embeddings storage

    def load_imgs(self, path: str) -> List[Image.Image]:
        """
        Returns a list of PIL images in a given path
        :param path:
        :return:
        """
        self.imgs_path = glob.glob(f'{path}/*')
        pil_imgs = [Image.open(img) for img in self.imgs_path]
        return pil_imgs

    def load_img_urls(self):
        """
        In case we want to load imgs from a list of urls
        :return:
        """
        pass

    def save_embs(self, pil_imgs: List[Image.Image]) -> None:
        """
        Extracts image embeddings from embedder and adds them to indexer
        :param pil_imgs:
        :return:
        """
        self.indexed_imgs_path.extend(self.imgs_path)
        img_embs = self.embedder.encode_imgs(pil_imgs)
        self.indexer.add(img_embs)

    def get_k_imgs(self, emb: np.ndarray, k: int):
        """
        Returns k indices of nearest image embeddings and respective distances for a given embedding emb
        :param emb:
        :param k:
        :return:
        """
        distances, indices = self.indexer.find(emb, k)
        return distances, np.array(self.indexed_imgs_path)[indices]

