import unittest
import numpy as np
from utils import accuracy
from ID3 import ID3

class Test(unittest.TestCase):

    def setUp(self):
        method = self._testMethodName
        print(method + f":{' ' * (25 - len(method))}", end='')

    def test_accuracy(self):
        y1 = np.array([1, 1, 1])  # (N1, D) = (3,)
        y2 = np.array([1, 0, 0])  # (N2, D) = (3,)

        accuracy_val = accuracy(y1, y2)
        self.assertEqual(accuracy_val, 1 / 3)
        print('Success')
    
    def test_entropy(self):
        rows = np.array([[0, 0, 1]]*19)
        labels = np.array([0]*9 + [1]*10)
        
        impurity = ID3.entropy(rows, labels)
        self.assertAlmostEqual(impurity, 0.99, delta=0.01)
        print('Success')


if __name__ == '__main__':
    unittest.main()
