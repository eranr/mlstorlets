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
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor

# TODO: Add here the serialization implementation
# 4 methods for now we import them.
from mlstorlets.utils.serialize_model import\
    classifier_from_string, classifier_to_string,\
    regressor_from_string, regressor_to_string,\
    deserialize_narray

class SGDEstimator(object):
    def __init__(self, logger):
        self.logger = logger

    def _read_data(self, in_file, params):
        try:
            num_features = int(params['num_features'])
            num_labels = int(params['num_labels'])
        except Exception as e:
            raise Exception(('Missing or Invalid num_features, num_labels') 
                            ('%s') % e)

        num_columns = num_features + num_labels
        T = np.loadtxt(in_file)
        if T.size % (num_features + num_labels) != 0:
            raise Exception(('Data size %d is not divide by the ')
                            ('number of columns %d') %
                            (T.size, num_features + num_labels))

        num_samples = T.size / (num_features + num_labels)
        T = np.reshape(T,(num_samples, num_columns))
        X, y, _junk = np.hsplit(T,np.array((num_features, num_columns)))
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
        self.logger.debug('Returning metadata')
        metadata = in_files[0].get_metadata()
        metadata['test'] = 'simple'
        out_files[0].set_metadata(metadata)

        self.logger.debug('Build Estimator')
        esttype = params['type']
        if esttype == 'SGDRegressor':
            estimator = regressor_from_string(params['serialized_estimator']) 
        elif esttype == 'SGDClassifier':
            estimator = classifier_from_string(params['serialized_estimator'])
        else:
            raise Exception('Unkown estimator type')

        X, y = self._read_data(in_files[0], params)

        command = params['command']
        sample_weight = self._get_array_param(params, 'sample_weight')
        if command == 'fit':
            coef_init = self._get_array_param(params, 'coef_init')
            intercept_init = self._get_array_param(params, 'intercept_init')
            estimator.fit(X,y, coef_init, intercept_init, sample_weight)
            if esttype == 'SGDRegressor':
                out_files[0].write(regressor_to_string(estimator))
            elif esttype == 'SGDClassifier':
                out_files[0].write(classifier_to_string(estimator))
        elif command == 'score':
            score = estimator.score(X, y, sample_weight)
            out_files[0].write(json.dumps({'score': score}))
        else:
            raise Exception('Unknown command %s' % command)


        self.logger.debug('Complete')
        in_files[0].close()
        out_files[0].close()
