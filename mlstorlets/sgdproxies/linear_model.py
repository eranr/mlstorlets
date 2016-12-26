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

import json

from sklearn.linear_model import SGDClassifier, SGDRegressor
from mlstorlets.utils.serialize_model import\
    classifier_to_string, classifier_from_string,\
    regressor_to_string, regressor_from_string,\
    serialize_narray
from mlstorlets.utils.swift_access import parse_config,\
    get_auth, invoke_storlet

class SGDProxyBase(object):
    def __init__(self, estimator, config_file):
        self.estimator = estimator
        if isinstance(estimator, SGDClassifier):
            self.type = 'SGDClassifier'
        if isinstance(estimator, SGDRegressor):
            self.type = 'SGDRegressor'

        self.serialize = classifier_to_string\
            if self.type == 'SGDClassifier' else regressor_to_string

        self.deserialize = classifier_from_string\
            if self.type == 'SGDClassifier' else regressor_from_string

        self.conf = None
        if config_file:
            self.conf = parse_config(config_file)

        self.url = self.token = None

    def _remote_command(self, data_url, command,
            num_features=None, num_labels=None,
            sample_weight=None,
            coef_init=None, intercept_init=None):

        if self.conf == None:
            raise Exception('No access config provided')

        if self.token == None:
            self.url, self.token = get_auth(self.conf)

        sest = self.serialize(self.estimator)

        ssample_weight = scoef_init = sintercept_init = None
        if sample_weight:
            ssample_weight = serialize_narray(sample_weight)
        if coef_init:
            scoef_init = serialize_narray(coef_init)
        if intercept_init:
            sintercept_init = serialize_narray(intercept_init)

        try:
            result = invoke_storlet(self.url, self.token,
                                    data_url,
                                    est_type=self.type,
                                    command=command,
                                    est=sest,
                                    num_features=num_features,
                                    num_labels=num_labels,
                                    sample_weight=ssample_weight,
                                    coef_init=scoef_init,
                                    intercept_init=sintercept_init)
        except Exception as e:
            # TODO: deal with auth exception to retry.
            pass
        return result

    def remote_fit(self, data_url,
            num_features=None, num_labels=None,
            coef_init=None, intercept_init=None,
            sample_weight=None):
        sest = self._remote_command(data_url, 'fit',
                                    num_features, num_labels,
                                    coef_init, intercept_init,
                                    sample_weight)
        self.estimator = self.deserialize(sest)        

    def simulate_remote_fit(self, X, y,
                            coef_init=None, intercept_init=None,
                            sample_weight=None):
        # Simulate storlet call:

        # Serialize internal classifier
        sest = self.serialize(self.estimator)

        # Simulate storlet side invocation
        est = self.deserialize(sest)
        est.fit(X,y)
        sest = self.serialize(est)

        # Update intenal state
        self.estimator = self.deserialize(sest)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        return self.estimator.fit(X, y,
                                  coef_init,
                                  intercept_init,
                                  sample_weight)

    def remote_score(self, data_url,
            num_features=None, num_labels=None,
            sample_weight=None):
        result = self._remote_command(data_url, 'score',
                                     num_features, num_labels,
                                     coef_init, intercept_init,
                                     sample_weight)
        return json.loads(result)['score']

    def simulate_remote_score(self, X, y,
                            sample_weight=None):
        # Simulate storlet call:

        # Serialize internal regressor
        sest = self.serialize(self.estimator)

        # Simulate storlet side invocation
        est = self.deserialize(sest)
        score = est.score(X, y, sample_weight)
        # in case a call to score does not change the internal state
        # the below serialization / de-serialization is redundant
        sest = self.serialize(est)

        # Update intenal state
        return score

    def decision_function(self, X):
        return self.estimator.decision_function(X)

    def densify(self):
        return self.estimator.densify()

    def sparsify(self):
        return self.estimator.sparsify()

    def get_params(self, deep=False):
        return self.estimator.get_params(deep)

    def set_params(self, *args, **kwargs):
        return self.estimator.set_params(args, kwargs)

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X)

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def intercept_(self):
        return self.estimator.intercept_

class SGDRegressorProxy(SGDProxyBase):

    def __init__(self, config_file=None,
            loss="squared_loss", penalty="l2", alpha=0.0001,
            l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
            verbose=0, epsilon=0.1, random_state=None,
            learning_rate="invscaling", eta0=0.01, power_t=0.25,
            warm_start=False, average=False):

        sgdregressor = SGDRegressor(
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

        super(SGDRegressorProxy, self).__init__(sgdregressor, config_file)

class SGDClassifierProxy(SGDProxyBase):

    def __init__(self, config_file=None,
            loss="hinge", penalty="l2", alpha=0.0001,
            l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
            verbose=0, epsilon=0.1, n_jobs=1, random_state=None,
            learning_rate="optimal", eta0=0.0, power_t=0.5,
            class_weight=None, warm_start=False, average=False):

        sgdclassifier = SGDClassifier(
            loss=loss, penalty=penalty,
            alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_iter=n_iter,shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0, power_t=power_t,
            class_weight=class_weight,
            warm_start=False,
            average=average)

        super(SGDClassifierProxy, self).__init__(sgdclassifier, config_file)
