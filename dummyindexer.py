"""
Created on 2022 Jan 31 18:38 

@author: keller
"""
import numpy as np
# from faiss import Indexer


class DummyIndexer():
    def __init__(self):
        """
        Creates an empty index object
        """
        self.index = None

    def add(self, embs: np.ndarray):
        """
        Adds new embeddings embs in empty or existing index
        :param embs:
        :return:
        """
        if self.index is None:
            self.index = embs
        else:
            self.index = np.append(self.index, embs, axis=0)

    def train(self):
        """
        Not sure if this one is necessary here, left for compatibility with abstract class Indexer
        :return:
        """
        pass

    def find(self, query: np.ndarray, topn: int) -> (np.ndarray, np.ndarray):
        """
        Returns topn entries closest to the query vector
        :param query:
        :param topn:
        :return:
        """
        similarities = (self.index @ query.squeeze())
        best_photo_idx = (-similarities).argsort()
        D, I = similarities[best_photo_idx[:topn]], best_photo_idx[:topn]
        return D, I

    def save(self, file: str):
        """
        Saves data to npy file
        :param file:
        :return:
        """
        np.save(file, self.index)

    def load(self, file: str):
        """
        Loads data from npy file
        :param file:
        :return:
        """
        self.index = np.load(file)

