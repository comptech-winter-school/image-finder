import unittest

from PIL import Image

from searchmodel import SearchModel

class TestSearchModel(unittest.TestCase):
    def setUp(self):
        self.sm = SearchModel(None, None)

    def test_load_imgs(self):
        pil_imgs = self.sm.load_imgs('../assets/test_pics')
        self.assertIsInstance(pil_imgs, list)
        self.assertIsInstance(pil_imgs[0], Image.Image)


if __name__ == '__main__':
    unittest.main()
