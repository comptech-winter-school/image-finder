"""
Created on 2022 Jan 28 14:09 

@author: keller
"""
import abc

import torch
import ruclip
import numpy as np

from PIL import Image

from numbers import Number
from typing import List

class Embedder(abc.ABC):
    @abc.abstractmethod
    def encode_text(self, text):
        pass

    @abc.abstractmethod
    def encode_imgs(self, imgs):
        pass

    def cos(self, emb1: np.ndarray, emb2: np.ndarray) -> Number:
        """
        Returns cos similarity between two embeddings
        :param emb1: 1D tensor
        :param emb2: 1D tensor
        :return: cos similarity (Number)
        """
        return np.dot(emb1, emb2) / np.linalg.norm(emb1, emb2)


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
        return text_latent.cpu().detach().numpy()

    def encode_imgs(self, pil_imgs: List[Image.Image]) -> np.ndarray:
        """
        Returns image latents of a image batch
        :param pil_imgs: list of PIL images
        :return img_latents: numpy array of img latents
        """
        with torch.no_grad():
            img_latents = self.predictor.get_image_latents(pil_imgs).cpu().detach().numpy()
        return img_latents
