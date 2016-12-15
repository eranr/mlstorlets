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

from mlstorlets.utils.serialize_model import\
    regressor_from_string, regressor_to_string
from sklearn.linear_model import SGDRegressor
from sklearn.datasets.samples_generator import make_blobs
from numpy import array_equal


class TestSGDRegressorSerialization(unittest.TestCase):

    def test_non_fitted(self):
        source_reg = SGDRegressor()
        serialized_reg = regressor_to_string(source_reg)
        target_reg = regressor_from_string(serialized_reg)
        print source_reg.coef_ is None
        print target_reg.coef_ is None
        self.assertEqual(source_reg.coef_, target_reg.coef_)
        self.assertEqual(source_reg.coef_, None)
        self.assertEqual(source_reg.t_, target_reg.t_)
        self.assertEqual(source_reg.t_, None)
        self.assertFalse(hasattr(target_reg,'intercept_'))
        self.assertFalse(hasattr(target_reg,'standard_coef_'))
        self.assertFalse(hasattr(target_reg,'standard_intecept_'))
        self.assertFalse(hasattr(target_reg,'average_coef_'))
        self.assertFalse(hasattr(target_reg,'average_intecept_'))
        self.assertFalse(hasattr(source_reg,'intercept_'))
        self.assertFalse(hasattr(source_reg,'standard_coef_'))
        self.assertFalse(hasattr(source_reg,'standard_intecept_'))
        self.assertFalse(hasattr(source_reg,'average_coef_'))
        self.assertFalse(hasattr(source_reg,'average_intecept_'))


    def test_avg_not_fitted(self):
        source_reg = SGDRegressor(average=True)
        serialized_reg = regressor_to_string(source_reg)
        target_reg = regressor_from_string(serialized_reg)
        self.assertEqual(source_reg.coef_, target_reg.coef_)
        self.assertEqual(source_reg.coef_, None)
        self.assertEqual(source_reg.standard_coef_, target_reg.standard_coef_)
        self.assertEqual(source_reg.standard_coef_, None)
        self.assertEqual(source_reg.average_coef_, target_reg.average_coef_)
        self.assertEqual(source_reg.average_coef_, None)
        self.assertEqual(source_reg.t_, target_reg.t_)
        self.assertEqual(source_reg.t_, None)
        self.assertFalse(hasattr(source_reg,'intercept_'))
        self.assertFalse(hasattr(target_reg,'intercept_'))
        self.assertFalse(hasattr(source_reg,'standard_intecept_'))
        self.assertFalse(hasattr(target_reg,'standard_intecept_'))
        self.assertFalse(hasattr(source_reg,'average_intecept_'))
        self.assertFalse(hasattr(target_reg,'average_intecept_'))

    def test_fitted(self):
        source_reg = SGDRegressor(average=False)
        # Do some fitting
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        source_reg.fit(X,Y)
        testX=[[1, 1], [0, 0]]
        source_reg.predict(testX)
        serialized_reg = regressor_to_string(source_reg)
        target_reg = regressor_from_string(serialized_reg)
        self.assertTrue(array_equal(source_reg.coef_, target_reg.coef_))
        self.assertFalse(source_reg.coef_ is None)
        self.assertTrue(array_equal(source_reg.intercept_,
                                    target_reg.intercept_))
        self.assertEqual(source_reg.t_, target_reg.t_)
        self.assertNotEqual(source_reg.t_, None)
        self.assertFalse(hasattr(source_reg,'standard_coef_'))
        self.assertFalse(hasattr(target_reg,'standard_coef_'))
        self.assertFalse(hasattr(source_reg,'average_coef_'))
        self.assertFalse(hasattr(target_reg,'average_coef_'))
        self.assertFalse(hasattr(source_reg,'standard_intecept_'))
        self.assertFalse(hasattr(target_reg,'standard_intecept_'))
        self.assertFalse(hasattr(target_reg,'average_intecept_'))
        self.assertFalse(hasattr(source_reg,'average_intecept_'))
        testX=[[1, 1], [0, 0]]
        source_res = source_reg.predict(testX)
        target_res = target_reg.predict(testX)
        self.assertTrue(array_equal(source_res, target_res))

    def test_avg_fitted(self):
        source_reg = SGDRegressor(average=True)
        # Do some fitting
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        source_reg.fit(X,Y)
        serialized_reg = regressor_to_string(source_reg)
        target_reg = regressor_from_string(serialized_reg)
        self.assertTrue(array_equal(source_reg.coef_, target_reg.coef_))
        self.assertFalse(source_reg.coef_ is None)
        self.assertTrue(array_equal(source_reg.intercept_,
                                    target_reg.intercept_))
        self.assertEqual(source_reg.t_, target_reg.t_)
        self.assertNotEqual(source_reg.t_, None)
        self.assertTrue(hasattr(source_reg,'standard_coef_'))
        self.assertTrue(hasattr(target_reg,'standard_coef_'))
        self.assertTrue(array_equal(source_reg.standard_coef_,
                                    target_reg.standard_coef_))
        self.assertTrue(hasattr(source_reg,'average_coef_'))
        self.assertTrue(hasattr(target_reg,'average_coef_'))
        self.assertTrue(array_equal(source_reg.average_coef_,
                                    target_reg.average_coef_))
        self.assertFalse(hasattr(source_reg,'standard_intecept_'))
        self.assertFalse(hasattr(target_reg,'standard_intecept_'))
        self.assertFalse(hasattr(target_reg,'average_intecept_'))
        self.assertFalse(hasattr(source_reg,'average_intecept_'))

        testX=[[1, 1], [0, 0]]
        source_res = source_reg.predict(testX)
        target_res = target_reg.predict(testX)
        self.assertTrue(array_equal(source_res, target_res))
