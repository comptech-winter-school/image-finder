"""
Created on 2022 Jan 28 14:09 

@author: keller
"""
import numpy as np
import torch
from numbers import Number
import ruclip
import abc
from PIL import Image
from typing import List

class Embedder(abc.ABC):
    @abc.abstractmethod
    def encode_text(self, text):
        pass

    @abc.abstractmethod
    def encode_imgs(self, imgs):
        pass

    def cos(self, text_emb: np.ndarray, img_emb: np.ndarray) -> Number:
        """
        Returns cos similarity between text_emb and img_emb
        :param text_emb: 1D tensor
        :param img_emb: 1D tensor
        :return: cos similarity (Number)
        """
        return np.dot(text_emb, img_emb)


class EmbedderRuCLIP(Embedder):
    def __init__(self, ruclip_model_name='ruclip-vit-base-patch32-384',
             device='cpu', templates = ['{}', 'это {}', 'на картинке {}']):
        """
        :param ruclip_model_name:
        :param device:
        :param templates:
        """
        clip, processor = ruclip.load(ruclip_model_name)
        self.predictor = ruclip.Predictor(clip, processor, device, bs=8, templates=templates)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Returns text latent of the text input
        :param text:
        :return:
        """
        classes = [text, ]
        with torch.no_grad():
            text_latent = self.predictor.get_text_latents(classes)
        return text_latent.detach().numpy()

    def encode_imgs(self, pil_imgs: List[Image.Image]) -> np.ndarray:
        """
        Returns image latents of a image batch
        :param pil_imgs: list of PIL images
        :return img_latents: numpy array of img latents
        """
        with torch.no_grad():
            img_latents = self.predictor.get_image_latents(pil_imgs).detach().numpy()
        return img_latents


### Tests
embedder = EmbedderRuCLIP()

text_emb = np.array([0, 1])
img_emb = np.array([0, 1])
print(embedder.cos(text_emb, img_emb))

text_emb = np.array([0, 1])
img_emb = np.array([1, 0])
print(embedder.cos(text_emb, img_emb))

text_emb = np.random.rand(10)
img_emb = np.random.rand(10)
print(embedder.cos(text_emb, img_emb))

text_emb = np.random.rand(512)
img_emb = np.random.rand(512)
print(embedder.cos(text_emb, img_emb))