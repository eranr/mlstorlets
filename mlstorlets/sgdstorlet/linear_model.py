# Copyright (c) 2016 itsonlyme.name
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
import StringIO
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor


# The following serialization finctions are copied from
#  mlstorlets/utils/serialize_model.py
# Ideally this should be a dependency, however, we currently do not
# have the appropriate dependency mechanism in place.
def serialize_narray(a):
    memfile = StringIO.StringIO()
    np.save(memfile, a)
    memfile.seek(0)
    return json.dumps(memfile.read().decode('latin-1'))


def deserialize_narray(sa):
    memfile = StringIO.StringIO()
    memfile.write(json.loads(sa).encode('latin-1'))
    memfile.seek(0)
    return np.load(memfile)


def estimator_to_string(est):
    params = est.get_params(deep=True)
    # coef_ can be either None or array
    params['coef_'] = None if est.coef_ is None\
        else serialize_narray(est.coef_)
    try:
        # intercept either exists or doesn't
        params['intercept_'] = serialize_narray(est.intercept_)
    except Exception as e:
        pass
    if est.average > 0:
        # average_coef_ can be either None or array
        params['average_coef_'] = None if est.average_coef_ is None\
            else serialize_narray(est.average_coef_)
        params['standard_coef_'] = None if est.standard_coef_ is None\
            else serialize_narray(est.standard_coef_)
        try:
            # average_intercept either exists or doesn't
            params['average_intercept_'] =\
                serialize_narray(est.average_intercept_)
            params['standard_intercept_'] =\
                serialize_narray(est.standard_intercept_)
        except Exception:
            pass

    params['t_'] = est.t_

    try:
        params['classes_'] = serialize_narray(est.classes_)
    except Exception as e:
        print e
        pass

    return json.dumps(params)


def _update_fitted_state(est, params):
    # coef_ can be either None or array
    est.coef_ = None if params['coef_'] is None\
        else deserialize_narray(params['coef_'])

    try:
        # intercept either exists or doesn't
        est.intercept_ = est.standard_intercept_ =\
            deserialize_narray(params['intercept_'])
    except Exception:
        pass

    if params['average'] > 0:
        # average_coef_ can be either None or array
        est.average_coef_ = None if params['average_coef_'] is None\
            else deserialize_narray(params['average_coef_'])
        est.standard_coef_ = None if params['standard_coef_'] is None\
            else deserialize_narray(params['standard_coef_'])

        try:
            # average_intercept either exists or doesn't
            est.average_intercept_ =\
                deserialize_narray(params['average_intercept_'])
            est.standard_intercept_ =\
                deserialize_narray(params['standard_intercept_'])
        except Exception:
            pass

    est.t_ = params['t_']

    try:
        est.classes_ = deserialize_narray(params['classes_'])
    except Exception:
        pass


def regressor_from_string(sest):
    params = json.loads(sest)
    regressor = SGDRegressor(
        loss=params['loss'], penalty=params['penalty'],
        alpha=params['alpha'],
        l1_ratio=params['l1_ratio'],
        fit_intercept=params['fit_intercept'],
        n_iter=params['n_iter'],
        shuffle=params['shuffle'],
        verbose=params['verbose'],
        epsilon=params['epsilon'],
        random_state=params['random_state'],
        learning_rate=params['learning_rate'],
        eta0=params['eta0'],
        power_t=params['power_t'],
        warm_start=params['warm_start'],
        average=params['average'])

    _update_fitted_state(regressor, params)
    return regressor


def classifier_from_string(sest):
    params = json.loads(sest)
    classifier = SGDClassifier(
        loss=params['loss'], penalty=params['penalty'],
        alpha=params['alpha'],
        l1_ratio=params['l1_ratio'],
        fit_intercept=params['fit_intercept'],
        n_iter=params['n_iter'],
        shuffle=params['shuffle'],
        verbose=params['verbose'],
        epsilon=params['epsilon'],
        n_jobs=params['n_jobs'],
        random_state=params['random_state'],
        learning_rate=params['learning_rate'],
        eta0=params['eta0'],
        power_t=params['power_t'],
        class_weight=params['class_weight'],
        warm_start=params['warm_start'],
        average=params['average'])

    _update_fitted_state(classifier, params)
    return classifier


def regressor_to_string(regressor):
    return estimator_to_string(regressor)


def classifier_to_string(classifer):
    return estimator_to_string(classifer)
# End of mlstorlets/utils/serialize_model.py


class SGDEstimator(object):
    def __init__(self, logger):
        self.logger = logger

    def _read_data(self, in_file, params, metadata):
        num_features = None
        try:
            num_features = int(params['num_features'])
        except Exception:
            pass
        try:
            num_features = int(metadata['Num-Features'])
        except Exception:
            pass

        num_labels = None
        try:
            num_labels = int(params['num_labels'])
        except Exception:
            pass
        try:
            num_labels = int(metadata['Num-Labels'])
        except Exception:
            pass

        if num_features is None or num_labels is None:
            raise Exception('Missing or Invalid num_features, num_labels')

        num_columns = num_features + num_labels
        T = np.loadtxt(in_file)
        if T.size % (num_features + num_labels) != 0:
            raise Exception(('Data size %d is not divide by the ')
                            ('number of columns %d') %
                            (T.size, num_features + num_labels))

        num_samples = T.size / (num_features + num_labels)
        T = np.reshape(T, (num_samples, num_columns))
        X, y, _junk = np.hsplit(T, np.array((num_features, num_columns)))
        return X, y

    def _get_array_param(self, params, param):
        try:
            array_param = deserialize_narray(params[param])
            return array_param
        except KeyError:
            return None
        except Exception:
            raise

    def __call__(self, in_files, out_files, params):
        """
        The function called for storlet invocation

        :param in_files: a list of StorletInputFile
        :param out_files: a list of StorletOutputFile
        :param params: a dict of request parameters
        """
        self.logger.debug('Returning metadata\n')
        metadata = in_files[0].get_metadata()
        self.logger.debug('Metadata is %s\n' % metadata)
        metadata['test'] = 'simple'
        out_files[0].set_metadata(metadata)

        self.logger.debug('Build Estimator\n')
        esttype = params['type']
        if esttype == 'SGDRegressor':
            estimator = regressor_from_string(params['serialized_estimator'])
        elif esttype == 'SGDClassifier':
            estimator = classifier_from_string(params['serialized_estimator'])
        else:
            raise Exception('Unkown estimator type')

        X, y = self._read_data(in_files[0], params, metadata)

        command = params['command']
        sample_weight = self._get_array_param(params, 'sample_weight')
        if command == 'fit':
            coef_init = self._get_array_param(params, 'coef_init')
            intercept_init = self._get_array_param(params, 'intercept_init')
            estimator.fit(X, y, coef_init, intercept_init, sample_weight)
            if esttype == 'SGDRegressor':
                out_files[0].write(regressor_to_string(estimator))
            elif esttype == 'SGDClassifier':
                out_files[0].write(classifier_to_string(estimator))
        elif command == 'score':
            score = estimator.score(X, y, sample_weight)
            out_files[0].write(json.dumps({'score': score}))
        else:
            raise Exception('Unknown command %s' % command)

        self.logger.debug('Complete\n')
        in_files[0].close()
        out_files[0].close()
