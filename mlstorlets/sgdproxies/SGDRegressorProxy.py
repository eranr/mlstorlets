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

from sklearn.linear_model import SGDRegressor
from mlstorlets.utils.serialize_model import regressor_to_string, regressor_from_string
import json


DEFAULT_EPSILON = 0.1
# Default value of ``epsilon`` parameter.


class SGDRegressorProxy(object):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
            l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
            verbose=0, epsilon=DEFAULT_EPSILON, random_state=None,
            learning_rate="invscaling", eta0=0.01, power_t=0.25,
            warm_start=False, average=False):

        self.sgdregressor = SGDRegressor(
            loss=loss, penalty=penalty,
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_iter=n_iter,shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0, power_t=power_t,
            warm_start=False,
            average=average)

    def remote_fit(self, path,
            coef_init=None, intercept_init=None,
            sample_weight=None):
        # Invokes the storlet and updates internal state of self.sgdregressor
        pass

    def simulate_remote_fit(self, X, y,
                            coef_init=None, intercept_init=None,
                            sample_weight=None):
        # Simulate storlet call:

        # Serialize internal regressor
        sreg = regressor_to_string(self.sgdregressor)

        # Simulate storlet side invocation
        regressor = regressor_from_string(sreg)
        regressor.fit(X,y)
        sreg = regressor_to_string(regressor)

        # Update intenal state
        self.sgdregressor = regressor_from_string(sreg)

    def remote_score(self, path,
            sample_weight=None):
        # Invokes the storlet and updates internal state of self.sgdregressor
        pass

    def simulate_remote_score(self, X, y,
                            sample_weight=None):
        # Simulate storlet call:

        # Serialize internal regressor
        sreg = regressor_to_string(self.sgdregressor)

        # Simulate storlet side invocation
        regressor = regressor_from_string(sreg)
        score = regressor.score(X, y, sample_weight)
        # in case a call to score does not change the internal state
        # the below serialization / de-serialization is redundant
        sreg = regressor_to_string(regressor)

        # Update intenal state
        self.sgdregressor = regressor_from_string(sreg)
        return score

    def remote_fit_transform(X, y=None, **fit_params):
        # Invokes the storlet fit_transform...
        pass
        
    def simulate_remote_fit_transform(self, X, y, **fit_params):
        # Serialize internal regressor
        sreg = regressor_to_string(self.sgdregressor)

        # Simulate storlet side invocation
        regressor = regressor_from_string(sreg)
        Xnew = regressor.fit_transform(X, y, **fit_params)
        sreg = regressor_to_string(regressor)

        # Update intenal state
        self.sgdregressor = regressor_from_string(sreg)
        return Xnew

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        return self.sgdregressor.fit(X, y,
                                  coef_init,
                                  intercept_init,
                                  sample_weight)

    def fit_transform(self, X, y, **fit_params):
        return self.sgdregressor.fit_transform(X, y, **fit_params)

    def densify(self):
        return self.sgdregressor.densify()

    def sparsify(self):
        return self.sgdregressor.sparsify()

    def get_params(self, deep=False):
        return self.sgdregressor.get_params(deep)

    def set_params(self, *args, **kwargs):
        return self.sgdregressor.set_params(args, kwargs)

    def predict(self, X):
        return self.sgdregressor.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.sgdregressor.score(X)

    def coef_(self):
        return self.sgdregressor.coef_

    def intercept_(self):
        return self.sgdregressor.intercept_

    def average_coef_(self):
        return self.sgdregressor.average_coef_

    def average_intercept_(self):
        return self.sgdregressor.average_intercept_
