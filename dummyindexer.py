"""
Created on 2022 Jan 31 18:38 

@author: keller
"""
import numpy as np
# from faiss import Indexer


class DummyIndexer():
    def __init__(self):
        self.index = None

    def add(self, embs: np.ndarray):
        if self.index is None:
            self.index = embs
        else:
            np.append(self.index, embs, axis=0)

    def train(self):
        pass

    def find(self, query: np.ndarray, topn: int):
        similarities = (self.index @ query.T())
        best_photo_idx = (-similarities).argsort()
        D, I = similarities[best_photo_idx[:topn]], best_photo_idx[:topn]
        return D, I

    def save(self, file):
        np.save(self.index)

    def load(self, file):
        self.index = np.load(file)