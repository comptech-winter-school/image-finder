
import pickle
import faissindexer
import abc



class Indexer(abc.ABC):

  @abc.abstractmethod
  def add(self, vs):
    pass
  
  @abc.abstractmethod
  def train(self, train_vectors):
    pass

  @abc.abstractmethod
  def find(self, query, topn): 
    pass
  
  @abc.abstractmethod
  def save(self,index):
    pass
    
  @abc.abstractmethod
  def load(self, file):
    pass



class FaissIndexer(Indexer):
  
  def __init__(self, dim, param, nprobe=5):
    self.index = faiss.index_factory(dim, param) # index creating, for more information: https://github.com/facebookresearch/faiss/wiki/The-index-factory
    self.dim = dim #dimention of vectors
    self.index.nprobe = nprobe #how many centroids we should pass
  
  def add(self, vs): # add vectors, which we used for train
    self.index.add(vs)
  
  def train(self, train_vectors):
    self.index.train(train_vectors) 

  def find(self, query, topn):  # D - distances; I - indices; query - test data; topn - the n nearest vectors
    D, I = self.index.search(query, topn) 
    return D, I
  
  def save(self,index):
    faiss.write_index(self.index, "flat.index")
   
  def load(self, file):
    self.index = faiss.read_index("file")

  
