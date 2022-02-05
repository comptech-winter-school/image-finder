import unittest

from PIL import Image
import numpy as np

from searchmodel import SearchModel
from dummyindexer import DummyIndexer
from embedder import EmbedderRuCLIP

class TestSearchModel(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbedderRuCLIP()
        self.indexer = DummyIndexer()
        self.sm = SearchModel(self.embedder, self.indexer)

    def test_init(self):
        self.assertFalse(self.sm.indexed_imgs_path)
        self.assertIsNone(self.sm.imgs_path)

    def test_load_imgs(self):
        pil_imgs = self.sm.load_imgs('../assets/test_pics')
        self.assertIsInstance(pil_imgs, list)
        self.assertIsInstance(pil_imgs[0], Image.Image)

    def test_save_embs(self):
        # index must be None before anything is added
        self.assertIsNone(self.indexer.index)
        pil_imgs = self.sm.load_imgs('../assets/test_pics')
        self.sm.save_embs(pil_imgs)
        # index must be non empty
        self.assertIsNotNone(self.indexer.index)
        self.assertEqual(len(self.sm.indexed_imgs_path), len(pil_imgs))

    def test_get_k_embs(self):
        pil_imgs = self.sm.load_imgs('../assets/test_pics')
        self.sm.save_embs(pil_imgs)

        query_emb = np.random.randn(1, 512)

        k = 0
        D, img_path = self.sm.get_k_imgs(query_emb, k)
        self.assertEqual(len(D), k)
        self.assertEqual(len(img_path), k)

        k = 1
        D, img_path = self.sm.get_k_imgs(query_emb, k)
        self.assertEqual(len(D), k)
        self.assertEqual(len(img_path), k)
        self.assertTrue(set(img_path).issubset(set(self.sm.indexed_imgs_path)))

        k = 5
        D, img_path = self.sm.get_k_imgs(query_emb, k)
        self.assertEqual(len(D), k)
        self.assertEqual(len(img_path), k)
        self.assertTrue(set(img_path).issubset(set(self.sm.indexed_imgs_path)))


if __name__ == '__main__':
    unittest.main()
