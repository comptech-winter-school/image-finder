import pytest
import numpy as np
from PIL import Image

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

    text_emb = np.array([0, 1])
    img_emb = np.array([0, 1])
    assert embedder.cos(text_emb, img_emb) == 1

    text_emb = np.array([0, 1, 0, 1, 3, 5, 6, 1])
    img_emb = np.array([7, 3, 5, 4, 2, 5, 9, 2])
    assert embedder.cos(text_emb, img_emb) == 94

    text_emb = np.array([0.1, 0.2, 0.3])
    img_emb = np.array([0.3, 0.2, 0.3])
    assert embedder.cos(text_emb, img_emb) == 0.16











