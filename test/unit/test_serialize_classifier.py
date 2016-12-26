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
    classifier_from_string, classifier_to_string
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
from numpy import array_equal


class TestSGDClassifierSerialization(unittest.TestCase):

    def test_non_fitted(self):
        source_cl = SGDClassifier()
        serialized_cl = classifier_to_string(source_cl)
        target_cl = classifier_from_string(serialized_cl)
        self.assertEqual(source_cl.coef_, target_cl.coef_)
        self.assertEqual(source_cl.coef_, None)
        self.assertEqual(source_cl.t_, target_cl.t_)
        self.assertEqual(source_cl.t_, None)
        self.assertFalse(hasattr(target_cl, 'intercept_'))
        self.assertFalse(hasattr(target_cl, 'standard_coef_'))
        self.assertFalse(hasattr(target_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(target_cl, 'average_coef_'))
        self.assertFalse(hasattr(target_cl, 'average_intecept_'))
        self.assertFalse(hasattr(source_cl, 'intercept_'))
        self.assertFalse(hasattr(source_cl, 'standard_coef_'))
        self.assertFalse(hasattr(source_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(source_cl, 'average_coef_'))
        self.assertFalse(hasattr(source_cl, 'average_intecept_'))

    def test_avg_not_fitted(self):
        source_cl = SGDClassifier(average=True)
        serialized_cl = classifier_to_string(source_cl)
        target_cl = classifier_from_string(serialized_cl)
        self.assertEqual(source_cl.coef_, target_cl.coef_)
        self.assertEqual(source_cl.coef_, None)
        self.assertEqual(source_cl.standard_coef_, target_cl.standard_coef_)
        self.assertEqual(source_cl.standard_coef_, None)
        self.assertEqual(source_cl.average_coef_, target_cl.average_coef_)
        self.assertEqual(source_cl.average_coef_, None)
        self.assertEqual(source_cl.t_, target_cl.t_)
        self.assertEqual(source_cl.t_, None)
        self.assertFalse(hasattr(source_cl, 'intercept_'))
        self.assertFalse(hasattr(target_cl, 'intercept_'))
        self.assertFalse(hasattr(source_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(target_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(source_cl, 'average_intecept_'))
        self.assertFalse(hasattr(target_cl, 'average_intecept_'))

    def test_fitted(self):
        source_cl = SGDClassifier(average=False)
        # Do some fitting
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        source_cl.fit(X, Y)
        testX = [[1, 1], [0, 0]]
        source_cl.predict(testX)
        serialized_cl = classifier_to_string(source_cl)
        target_cl = classifier_from_string(serialized_cl)
        self.assertTrue(array_equal(source_cl.coef_, target_cl.coef_))
        self.assertFalse(source_cl.coef_ is None)
        self.assertTrue(array_equal(source_cl.intercept_,
                                    target_cl.intercept_))
        self.assertEqual(source_cl.t_, target_cl.t_)
        self.assertNotEqual(source_cl.t_, None)
        self.assertFalse(hasattr(source_cl, 'standard_coef_'))
        self.assertFalse(hasattr(target_cl, 'standard_coef_'))
        self.assertFalse(hasattr(source_cl, 'average_coef_'))
        self.assertFalse(hasattr(target_cl, 'average_coef_'))
        self.assertFalse(hasattr(source_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(target_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(target_cl, 'average_intecept_'))
        self.assertFalse(hasattr(source_cl, 'average_intecept_'))
        testX = [[1, 1], [0, 0]]
        source_res = source_cl.predict(testX)
        target_res = target_cl.predict(testX)
        self.assertTrue(array_equal(source_res, target_res))

    def test_avg_fitted(self):
        source_cl = SGDClassifier(average=True)
        # Do some fitting
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        source_cl.fit(X, Y)
        print source_cl.classes_
        serialized_cl = classifier_to_string(source_cl)
        print serialized_cl
        target_cl = classifier_from_string(serialized_cl)
        print target_cl.classes_
        self.assertTrue(array_equal(source_cl.coef_, target_cl.coef_))
        self.assertFalse(source_cl.coef_ is None)
        self.assertTrue(array_equal(source_cl.intercept_,
                                    target_cl.intercept_))
        self.assertEqual(source_cl.t_, target_cl.t_)
        self.assertNotEqual(source_cl.t_, None)
        self.assertTrue(hasattr(source_cl, 'standard_coef_'))
        self.assertTrue(hasattr(target_cl, 'standard_coef_'))
        self.assertTrue(array_equal(source_cl.standard_coef_,
                                    target_cl.standard_coef_))
        self.assertTrue(hasattr(source_cl, 'average_coef_'))
        self.assertTrue(hasattr(target_cl, 'average_coef_'))
        self.assertTrue(array_equal(source_cl.average_coef_,
                                    target_cl.average_coef_))
        self.assertFalse(hasattr(source_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(target_cl, 'standard_intecept_'))
        self.assertFalse(hasattr(target_cl, 'average_intecept_'))
        self.assertFalse(hasattr(source_cl, 'average_intecept_'))

        testX = [[1, 1], [0, 0]]
        source_res = source_cl.predict(testX)
        target_res = target_cl.predict(testX)
        self.assertTrue(array_equal(source_res, target_res))
