import pytest
import numpy as np

from embedder import EmbedderRuCLIP

def test_encode_text():
    embedder = EmbedderRuCLIP()
    text = "На картинке котик"
    assert embedder.encode_text(text)

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











