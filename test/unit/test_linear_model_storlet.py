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

import os
import json
import unittest

from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import SGDRegressor, SGDClassifier
import numpy as np

from storlets.agent.daemon import files

from mlstorlets.utils.serialize_model import\
    classifier_from_string, classifier_to_string,\
    regressor_from_string, regressor_to_string
from mlstorlets.sgdstorlet.linear_model import SGDEstimator
from test import data_file_create, data_storlet_file_open,\
    data_file_read, data_file_destroy
from test.unit import FakeLogger


class TestLinearModelStorlet(unittest.TestCase):

    def setUp(self):
        self.filename = '/tmp/blobs.txt'

    def tearDown(self):
        data_file_destroy(self.filename)

    def _get_local_est(self, est_type):
        return SGDRegressor(shuffle=False) if est_type == 'SGDRegressor'\
            else SGDClassifier(shuffle=False)

    def _prepare_storlet_input_file(self):
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        data_file_create(self.filename, X, Y)
        return data_storlet_file_open(self.filename)

    def _invoke_sgdestimator(self, local_est, esttype, command,
                             num_features, num_labels):
        md_infd, md_outfd = os.pipe()
        obj_infd, obj_outfd = os.pipe()

        out_files = [files.StorletOutputFile(md_outfd, obj_outfd)]
        in_files = [self._prepare_storlet_input_file()]
        sest = regressor_to_string(local_est) if esttype == 'SGDRegressor'\
            else classifier_to_string(local_est)
        params = {'type': esttype,
                  'command': command,
                  'num_features': num_features,
                  'num_labels': num_labels,
                  'serialized_estimator': sest}
        sdgest = SGDEstimator(logger=FakeLogger())
        sdgest(in_files, out_files, params)
        out_md = json.loads(os.read(md_infd, 100))
        if command == 'fit':
            sest = os.read(obj_infd, 1024)
            est = regressor_from_string(sest) if esttype == 'SGDRegressor'\
                else classifier_from_string(sest)
            return est
        elif command == 'score':
            sscore = os.read(obj_infd, 1024)
            dscore = json.loads(sscore)
            return dscore['score']

    def _calculate_z(self, est):
        xx = np.linspace(-1, 5, 10)
        yy = np.linspace(-1, 5, 10)
        X1, X2 = np.meshgrid(xx, yy)
        Z = np.empty(X1.shape)
        for (i, j), val in np.ndenumerate(X1):
            x1 = val
            x2 = X2[i, j]
            p = est.decision_function([[x1, x2]])
            Z[i, j] = p[0]
        return Z

    def test_regressor_fit(self):
        local_est = SGDRegressor(shuffle=False)
        regressor = self._invoke_sgdestimator(local_est,
                                              'SGDRegressor', 'fit',
                                              '2', '1')
        Xtest = [[0.5, 0.7], [1, 2]]
        p1 = regressor.predict(Xtest)

        regressor = SGDRegressor(shuffle=False)
        X, Y = data_file_read(self.filename, 2, 1)
        regressor.fit(X, Y)
        p2 = regressor.predict(Xtest)
        self.assertTrue(np.array_equal(p1, p2))

    def test_classifier_fit(self):
        local_est = SGDClassifier(shuffle=False)
        classifier = self._invoke_sgdestimator(local_est,
                                               'SGDClassifier', 'fit',
                                               '2', '1')
        Z1 = self._calculate_z(classifier)

        classifier = SGDClassifier(shuffle=False)
        X, Y = data_file_read(self.filename, 2, 1)
        classifier.fit(X, Y)
        Z2 = self._calculate_z(classifier)
        self.assertTrue(np.array_equal(Z1, Z2))

    def _test_score(self, est_type):
        Xt, Yt = make_blobs(n_samples=1000, centers=2,
                            random_state=1, cluster_std=0.60)
        local_est = self._get_local_est(est_type)
        local_est.fit(Xt, Yt)
        s1 = self._invoke_sgdestimator(local_est,
                                       est_type, 'score',
                                       '2', '1')

        local_est = self._get_local_est(est_type)
        local_est.fit(Xt, Yt)
        X, Y = data_file_read(self.filename, 2, 1)
        s2 = local_est.score(X, Y)

        self.assertEqual(s1, s2)

    def test_regressor_score(self):
        self._test_score('SGDRegressor')

    def test_classifier_score(self):
        self._test_score('SGDClassifier')

# TODO: Add tests:
#       for fit with init params and weight
#       for score with weight
