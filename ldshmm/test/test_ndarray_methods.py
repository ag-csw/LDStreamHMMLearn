from unittest import TestCase

import numpy as np



class TestNdArrayMethods(TestCase):

    def setUp(self):
        self.data = np.arange(50).reshape(5,10)

    def test_split (self):
        dataslice = []
        num=3
        for i in range(0, num):
            dataslice.append(self.data[i, :])

        flat = self.data.flatten()
        print("Flattened Array", flat)
        dataslice2 = np.split(flat,num)
        print(dataslice)
        print(dataslice2)
        np.testing.assert_array_equal(dataslice,dataslice2)