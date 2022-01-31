"""
Created on 2022 Jan 30 10:14 

@author: keller
"""
import glob
from PIL import Image
from typing import List


class SearchModel():
    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer

    def load_imgs(self, path: str) -> List[Image.Image]:
        """
        Returns a list of PIL images in a given path
        :param path:
        :return:
        """
        imgs = glob.glob(f'{path}/*')
        pil_imgs = [Image.open(img) for img in imgs]
        return pil_imgs

    def get_embs(self, path):
        """
        Receives a list of PIL images and return their CLIP embeddings
        :param path:
        :return:
        """
        pil_imgs = self.load_imgs(path)
        img_embs = self.embedder.encode_imgs(pil_imgs)
        del pil_imgs
        return img_embs

    def load_embs(self, path):
        img_embs = self.get_embs(path)
        self.indexer.add(img_embs)

    def get_k_imgs(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


sm = SearchModel(None, None)
sm.load_imgs('assets/pics')
