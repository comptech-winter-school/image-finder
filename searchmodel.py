"""
Created on 2022 Jan 30 10:14 

@author: keller
"""
import glob


class SearchModel():
    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer

    def load_imgs(self, path):
        imgs = glob.glob(f'{path}/*')
        print(imgs)

    def get_embeddings(self):
        pass

    def load_embeddings(self):
        pass

    def get_k_imgs(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


sm = SearchModel(None, None)
sm.load_imgs('assets/pics')
