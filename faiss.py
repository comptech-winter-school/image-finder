import pickle
import faiss
import abc

class FaissIndexer:
  
  @abc.abstractmethod
  def __init__(self, dim, param, nprobe=5):
    self.index = faiss.index_factory(dim, param) # index creating, for more information: https://github.com/facebookresearch/faiss/wiki/The-index-factory
    self.dim = dim #dimention of vectors
    self.index.nprobe = nprobe #how many centroids we should pass
  
  
  def add(self, vs):
    self.index.add(vs)
  
  @abc.abstractmethod
  def train(self, train_vectors):
    self.index.train(train_vectors) 

  def find(self, query, topn):
    D, I = self.index.search(query, topn) 
    return D, I

  def save(self,index):
    faiss.write_index(self.index, "flat.index")
    '''
    f = open(file, 'wb')
    pickle.dump(self.index, file)
    f.close()
    '''

  def load(self, file):
    self.index = faiss.read_index("file")
    '''
    f = open(file, 'rb')
    self.index = pickle.load(f)
    f.close()
    '''
