"""
Created on 2022 Jan 28 14:09 

@author: keller
"""

import ruclip
import abc
import torch


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

    def cos(self, text_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        """
        Returns cos similarity between text_emb and img_emb
        :param text_emb: tensor, shape 1x512
        :param img_emb: tensor, shape 1x512
        :return: cos similarity
        """
        return torch.dot(text_emb, img_emb)
