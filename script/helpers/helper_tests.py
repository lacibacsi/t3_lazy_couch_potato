#! /usr/bin/env python

import unittest
import numpy as np
import helper_methods


class HelperTests(unittest.TestCase):

    def mean_test(self):
        '''
            Testing mean calculation of numpy array slicing
            The meathod is used in LIDAR mean obstalce distance calculation
        '''
        values = [2, 3, 4, 0, 0, 0, 0, 0, 2, 3, 4]
        nparray = np.array(values)
        print(nparray)
        mean = helper_methods.MeanOfArraysTwoEnd(nparray, 3)
        self.assertEqual(3.0, mean)
