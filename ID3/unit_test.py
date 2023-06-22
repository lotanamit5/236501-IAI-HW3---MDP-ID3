import unittest
import numpy as np
from utils import accuracy


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



if __name__ == '__main__':
    unittest.main()
