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

from sklearn.linear_model import SGDClassifier
from mlstorlets.sgdproxies.SGDClassifierProxy  import SGDClassifierProxy
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


class TestSDGClassifierProxy(unittest.TestCase):

    def _iter_minibatches(self, X, Y, chunksize):
        # Provide chunks one by one
        chunkstartmarker = 0
        numtrainingpoints = 1000
        while chunkstartmarker < numtrainingpoints:
            X_chunk = X[chunkstartmarker:chunkstartmarker + chunksize]
            y_chunk = Y[chunkstartmarker:chunkstartmarker + chunksize]
            yield X_chunk, y_chunk
            chunkstartmarker += chunksize

    def _fit_iter(self, est, X, Y):
        batcherator = self._iter_minibatches(X, Y, chunksize=10)
        # Train model
        for X_chunk, y_chunk in batcherator:
            if isinstance(est, SGDClassifier):
                est.fit(X_chunk, y_chunk)
            else:
                est.simulate_remote_fit(X_chunk, y_chunk)


    def _predict(self, est, testX):
        # Now make predictions with trained model
        y_predicted = est.predict(testX)
        return y_predicted
        

    def test_basic_func(self):
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        Xtest = [[0.5, 0.7], [1, 2]]
        model = SGDClassifier(n_iter=10000, epsilon=0.0001, shuffle=False)
        self._fit_iter(model, X, Y)
        p1 = self._predict(model, Xtest)

        model = SGDClassifierProxy(n_iter=10000, epsilon=0.0001, shuffle=False)
        self._fit_iter(model, X, Y)
        p2 = self._predict(model, Xtest)

        self.assertTrue(np.array_equal(p1, p2))

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

    def test_max_margin_sep(self):
        X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

        # fit the model
        model = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200,
                              fit_intercept=True, shuffle=False)
        model.fit(X, Y)
        Z1 = self._calculate_z(model)


        model = SGDClassifierProxy(loss="hinge", alpha=0.01, n_iter=200,
                                   fit_intercept=True, shuffle=False)
        model.simulate_remote_fit(X, Y)
        Z2 = self._calculate_z(model)
        self.assertTrue(np.array_equal(Z1, Z2))

    def test_fit_transform(self):
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        model = SGDClassifier(n_iter=10000, epsilon=0.0001, shuffle=False)
        Xnew1 = model.fit_transform(X, Y)
        model = SGDClassifierProxy(n_iter=10000, epsilon=0.0001, shuffle=False)
        Xnew2 = model.simulate_remote_fit_transform(X, Y)
        self.assertTrue(np.array_equal(Xnew1, Xnew2))

    def _get_fitted_model(self, proxy=True):
        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=0, cluster_std=0.60)
        if proxy == False:
            model = SGDClassifier(n_iter=10000, epsilon=0.0001, shuffle=False)
            model.fit(X, Y)
        else:
            model = SGDClassifierProxy(n_iter=10000, epsilon=0.0001, shuffle=False)
            model.simulate_remote_fit(X, Y)

        return model

    def test_sparsify(self):
        model = self._get_fitted_model(proxy=False)
        s1 = model.sparsify().coef_
        model = self._get_fitted_model(proxy=True)
        s2 = model.sparsify().coef_
        self.assertTrue((s1!=s2).nnz==0)

    def test_densify(self):
        model = self._get_fitted_model(proxy=False)
        s1 = model.sparsify().densify().coef_
        model = self._get_fitted_model(proxy=True)
        s2 = model.sparsify().densify().coef_
        self.assertTrue(np.array_equal(s1, s2))
