import unittest
import numpy as np
from dummyindexer import DummyIndexer


class TestDummyIndexer(unittest.TestCase):
    def setUp(self):
        self.dummy_indexer = DummyIndexer()

    def test_init(self):
        self.assertIsNone(self.dummy_indexer.index)

    def test_add(self):
        shape = (10, 512)
        embs = np.random.randn(*shape)
        self.dummy_indexer.add(embs)
        self.assertEqual(self.dummy_indexer.index.shape, shape)

        self.dummy_indexer.add(embs)
        self.assertEqual(self.dummy_indexer.index.shape, (2 * shape[0], shape[1]))

    def test_find(self):
        shape = (10, 512)
        embs = np.random.randn(*shape)
        self.dummy_indexer.add(embs)

        dim = 512
        query_emb = np.random.randn(dim)

        topn = 0
        D, I = self.dummy_indexer.find(query_emb, topn)
        self.assertEqual(len(D), topn)
        self.assertEqual(len(I), topn)

        topn = 1
        D, I = self.dummy_indexer.find(query_emb, topn)
        self.assertEqual(len(D), topn)
        self.assertEqual(len(I), topn)

        topn = 3
        D, I = self.dummy_indexer.find(query_emb, topn)
        self.assertEqual(len(D), topn)
        self.assertEqual(len(I), topn)

        topn = 10
        D, I = self.dummy_indexer.find(query_emb, topn)
        self.assertEqual(len(D), topn)
        self.assertEqual(len(I), topn)

        # If topn is larger than total number of entries in index, find should return all entries
        topn = 15
        D, I = self.dummy_indexer.find(query_emb, topn)
        self.assertEqual(len(D), shape[0])
        self.assertEqual(len(I), shape[0])

    def test_save(self):
        shape = (10, 512)
        embs = np.random.randn(*shape)
        self.dummy_indexer.add(embs)
        self.dummy_indexer.save('test_dummy.npy')

    def test_load(self):
        shape = (10, 512)
        embs = np.random.randn(*shape)
        self.dummy_indexer.add(embs)
        self.dummy_indexer.load('test_dummy.npy')


if __name__ == '__main__':
    unittest.main()
