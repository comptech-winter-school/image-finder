import pickle
import faiss
import abc

class FaissIndexer:
  
  @abc.abstractmethod
  def __init__(self, dim, param, nprobe=5):
    self.index = faiss.index_factory(dim, param)
    self.dim = dim
    self.index.nprobe = nprobe
  
  
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
