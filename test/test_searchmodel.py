import unittest

from searchmodel import SearchModel

class TestSearchModel(unittest.TestCase):
    def setUp(self):
        self.sm = SearchModel(None, None)

    def test_load_imgs(self):
        self.sm.load_imgs('assets/pics')


if __name__ == '__main__':
    unittest.main()
