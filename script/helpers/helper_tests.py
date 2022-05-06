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
        values = [2, 3, 4, 0, 0, 0, 0, 0, 4, 6, 8]
        nparray = np.array(values)
        print(nparray)
        mean = helper.MeanOfArraysTwoEnd(nparray, 3)
        self.assertEqual(4.5, mean)
