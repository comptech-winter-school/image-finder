import unittest
import numpy as np
from PIL import Image
from scipy.spatial import distance

from embedder import EmbedderRuCLIP


class TestEmbedderRuCLIP(unittest.TestCase):
    def test_encode_text(self):
        embedder = EmbedderRuCLIP()
        text = "На картинке котик"
        text_emb = embedder.encode_text(text)
        self.assertEqual(text_emb.shape, (1, 512))
        self.assertTrue(np.isclose(np.linalg.norm(text_emb), 1))

    def test_encode_imgs(self):
        embedder = EmbedderRuCLIP()
        N = 1
        pil_imgs = [Image.fromarray(
            (255 * np.random.randn(256, 256, 3)).astype(np.uint8)) for _ in range(N)]
        img_embs = embedder.encode_imgs(pil_imgs)
        self.assertEqual(img_embs.shape, (N, 512))

        N = 100
        pil_imgs = [Image.fromarray(
            (255 * np.random.randn(256, 256, 3)).astype(np.uint8)) for _ in range(N)]
        img_embs = embedder.encode_imgs(pil_imgs)
        self.assertEqual(img_embs.shape, (N, 512))

    def test_cos(self):
        embedder = EmbedderRuCLIP()

        emb1 = np.array([0, 1])
        emb2 = np.array([0, 1])
        self.assertAlmostEqual(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))

        emb1 = np.array(np.random.rand(1, 10))
        emb2 = np.array(np.random.rand(1, 10))
        self.assertAlmostEqual(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))

        emb1 = np.array(np.random.rand(1, 512))
        emb2 = np.array(np.random.rand(1, 512))
        self.assertAlmostEqual(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))

        emb1 = np.array(np.random.rand(512))
        emb2 = np.array(np.random.rand(512))
        self.assertAlmostEqual(embedder.cos(emb1, emb2), (1 - distance.cosine(emb1, emb2)))


if __name__ == '__main__':
    unittest.main()








