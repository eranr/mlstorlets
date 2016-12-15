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

from sklearn.linear_model import SGDClassifier, SGDRegressor
from mlstorlets.utils.serialize_model import\
    classifier_to_string, classifier_from_string,\
    regressor_to_string, regressor_from_string
import json

class SGDProxyBase(object):
    def __init__(self, estimator):
        self.estimator = estimator
        if isinstance(estimator, SGDClassifier):
            self.type = 'SGDClassifier'
        if isinstance(estimator, SGDRegressor):
            self.type = 'SGDRegressor'

        self.serialize = classifier_to_string\
            if self.type == 'SGDClassifier' else regressor_to_string

        self.deserialize = classifier_from_string\
            if self.type == 'SGDClassifier' else regressor_from_string

    def remote_fit(self, path,
            coef_init=None, intercept_init=None,
            sample_weight=None):
        # Invokes the storlet and updates internal state of self.sgdclassifier
        pass

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
        self.estimator = self.deserialize(sest)
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

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
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

        super(SGDRegressorProxy, self).__init__(sgdregressor)

class SGDClassifierProxy(SGDProxyBase):

    def __init__(self, loss="hinge", penalty="l2", alpha=0.0001,
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

        super(SGDClassifierProxy, self).__init__(sgdclassifier)

#    def remote_fit(self, path,
#            coef_init=None, intercept_init=None,
#            sample_weight=None):
#        # Invokes the storlet and updates internal state of self.sgdclassifier
#        pass
#
#    def simulate_remote_fit(self, X, y,
#                            coef_init=None, intercept_init=None,
#                            sample_weight=None):
#        # Simulate storlet call:
#
#        # Serialize internal classifier
#        sreg = classifier_to_string(self.sgdclassifier)
#
#        # Simulate storlet side invocation
#        classifier = classifier_from_string(sreg)
#        classifier.fit(X,y)
#        sreg = classifier_to_string(classifier)
#
#        # Update intenal state
#        self.sgdclassifier = classifier_from_string(sreg)
#
#    def fit(self, X, y, coef_init=None, intercept_init=None,
#            sample_weight=None):
#        return self.sgdclassifier.fit(X, y,
#                                  coef_init,
#                                  intercept_init,
#                                  sample_weight)
#
#    def fit_transform(self, X, y, **fit_params):
#        return self.sgdclassifier.fit_transform(X, y, **fit_params)
#
#    def remote_score(self, path,
#            sample_weight=None):
#        # Invokes the storlet and updates internal state of self.sgdregressor
#        pass
#
#    def simulate_remote_score(self, X, y,
#                            sample_weight=None):
#        # Simulate storlet call:
#
#        # Serialize internal regressor
#        scli = classifier_to_string(self.sgdclassifier)
#
#        # Simulate storlet side invocation
#        classifier = classifier_from_string(scli)
#        score = classifier.score(X, y, sample_weight)
#        # in case a call to score does not change the internal state
#        # the below serialization / de-serialization is redundant
#        scli = classifier_to_string(classifier)
#
#        # Update intenal state
#        self.classifier = classifier_from_string(scli)
#        return score
#
#    def decision_function(self, X):
#        return self.sgdclassifier.decision_function(X)
#
#    def densify(self):
#        return self.sgdclassifier.densify()
#
#    def sparsify(self):
#        return self.sgdclassifier.sparsify()
#
#    def get_params(self, deep=False):
#        return self.sgdclassifier.get_params(deep)
#
#    def set_params(self, *args, **kwargs):
#        return self.sgdclassifier.set_params(args, kwargs)
#
#    def predict(self, X):
#        return self.sgdclassifier.predict(X)
#
#    def score(self, X, y, sample_weight=None):
#        return self.sgdclassifier.score(X)
#
#    @property
#    def coef_(self):
#        return self.sgdclassifier.coef_
#
#    @property
#    def intercept_(self):
#        return self.sgdclassifier.intercept_
