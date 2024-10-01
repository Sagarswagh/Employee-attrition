import unittest
import os
from ml_code.data_load import load_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        cwd = os.getcwd()
        data = load_data(cwd + "/data/data.csv")
        self.assertIsNotNone(data)
        self.assertEqual(len(data),1470)

if __name__ == 'main':
    unittest.main()