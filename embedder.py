"""
Created on 2022 Jan 28 14:09 

@author: keller
"""
import abc

class Embedder(abc.ABC):
    @abc.abstractmethod
    def encode_text(self, text):
        pass

   @abc.abstractmethod
    def encode_img(self, img):
        pass

   @abc.abstractmethod
    def encode_imgs(self, imgs):
        pass

   @abc.abstractmethod
   def cos(a, b):
       pass