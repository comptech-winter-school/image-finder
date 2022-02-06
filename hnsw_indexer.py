import numpy as np
import nmslib

class HnmsIndexer:
    def __init__(self):
        """
        Creates an empty index object
        """
        self.index = nmslib.init(method='hnsw', space='cosinesimil')

    def add(self, embs: np.ndarray):
        """
        Adds new embeddings embs in empty or existing index
        :param embs:
        :return:
        """
        self.index.addDataPointBatch(embs)
        self.index.createIndex({'post': 2}, print_progress=True)


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
        I, D = self.index.knnQuery(query, k=topn)
        return 1-D, I

    def save(self, file: str):
        """
        Saves data to npy file
        :param file:
        :return:
        """
        self.index.saveIndex(file)

    def load(self, file: str):
        """
        Loads data from npy file
        :param file:
        :return:
        """
        self.index.loadIndex(file)

