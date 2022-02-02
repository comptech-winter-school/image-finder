"""
Created on 2022 Jan 28 14:09 
@author: keller
"""
import abc

import torch
import ruclip
import clip
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
        emb1, emb2 = emb1.squeeze(), emb2.squeeze() # convert (1, N) arrays to (N,)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


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

    def _tonumpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Detaches tensor from GPU and converts it to numpy array
        :return: numpy array
        """
        return tensor.cpu().detach().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Returns text latent of the text input
        :param text:
        :return:
        """
        classes = [text, ]
        with torch.no_grad():
            text_latent = self.predictor.get_text_latents(classes)
        return self._tonumpy(text_latent)

    def encode_imgs(self, pil_imgs: List[Image.Image]) -> np.ndarray:
        """
        Returns image latents of a image batch
        :param pil_imgs: list of PIL images
        :return img_latents: numpy array of img latents
        """
        with torch.no_grad():
            img_latents = self.predictor.get_image_latents(pil_imgs)
        return self._tonumpy(img_latents)

class EmbedderCLIP(Embedder):
    def __init__(self, clip_model_name='ViT-B/32', device='cpu'):
        """
        :param clip_model_name:
        :param device:
        """
        self.device = device
        self.predictor, self.preprocess = clip.load(clip_model_name, device=device)

    def _tonumpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Detaches tensor from GPU and converts it to numpy array
        :return: numpy array
        """
        return tensor.cpu().detach().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Returns text latent of the text input
        :param text:
        :return:
        """
        with torch.no_grad():
          # Encode it to a feature vector using CLIP
          text_latent = self.predictor.encode_text(clip.tokenize(text).to(self.device))
          text_latent /= text_latent.norm(dim=-1, keepdim=True)
          
        return self._tonumpy(text_latent)

    def encode_imgs(self, pil_imgs: List[Image.Image]) -> np.ndarray:
        """
        Returns image latents of a image batch
        :param pil_imgs: list of PIL images
        :return img_latents: numpy array of img latents
        """

        # Preprocess all photos
        photos_preprocessed = torch.stack([self.preprocess(photo) for photo in pil_imgs]).to(self.device)

        with torch.no_grad():
          # Encode the photos batch to compute the feature vectors and normalize them
          img_latents = self.predictor.encode_image(photos_preprocessed)
          img_latents /= img_latents.norm(dim=-1, keepdim=True)

        return self._tonumpy(img_latents)