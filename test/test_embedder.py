import unittest
import numpy as np
from PIL import Image
from scipy.spatial import distance

from embedder import EmbedderRuCLIP


def test_encode_text():
    embedder = EmbedderRuCLIP()
    text = "На картинке котик"
    assert embedder.encode_text(text).shape == (1, 512)
    assert np.isclose(np.linalg.norm(embedder.encode_text(text)), 1)

def test_encode_imgs():
    embedder = EmbedderRuCLIP()
    N = 1
    pil_imgs = [Image.fromarray(
        (255 * np.random.randn(256, 256, 3)).astype(np.uint8)) for _ in range(N)]
    assert embedder.encode_imgs(pil_imgs).shape == (N, 512)

    N = 100
    pil_imgs = [Image.fromarray(
        (255 * np.random.randn(256, 256, 3)).astype(np.uint8)) for _ in range(N)]
    assert embedder.encode_imgs(pil_imgs).shape == (N, 512)


def test_cos():
    embedder = EmbedderRuCLIP()

    emb1 = np.array([0, 1])
    emb2 = np.array([0, 1])
    assert np.isclose(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))

    emb1 = np.array(np.random.rand(1, 10))
    emb2 = np.array(np.random.rand(1, 10))
    assert np.isclose(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))

    emb1 = np.array(np.random.rand(1, 512))
    emb2 = np.array(np.random.rand(1, 512))
    assert np.isclose(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))

    emb1 = np.array(np.random.rand(512))
    emb2 = np.array(np.random.rand(512))
    assert np.isclose(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))











