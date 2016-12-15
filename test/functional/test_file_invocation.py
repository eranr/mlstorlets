# Copyright (c) 2015-2016 itsonlyme.name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.datasets.samples_generator import make_blobs

from test import data_file_create,\
    data_file_read, data_file_destroy


class TestConsumptionFromFile(unittest.TestCase):

    def setUp(self):
        self.X, self.Y = make_blobs(n_samples=1000, centers=2,
                                    random_state=0, cluster_std=0.60)
        data_file_create('/tmp/blobs.txt', self.X, self.Y)
        self.x1, self.y1 = data_file_read('/tmp/blobs.txt', 2, 1)

    def tearDown(self):
        data_file_destroy('/tmp/blobs.txt')

    def test_arrays_compare(self):
        self.assertTrue(np.array_equal(self.X, self.x1))
        self.assertTrue(np.array_equal(self.Y, self.y1))

    def test_fit(self):
        regressor = SGDRegressor(shuffle=False)
        regressor.fit(self.X, self.Y)
        Xtest = [[0.5, 0.7], [1, 2]]
        p1 = regressor.predict(Xtest)

        regressor = SGDRegressor(shuffle=False)
        regressor.fit(self.x1, self.y1)
        p2 = regressor.predict(Xtest)
        self.assertTrue(np.array_equal(p1, p2))
