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
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor

from storlets.agent.daemon import files
from mlstorlets.utils.serialize_model import\
    classifier_from_string, regressor_from_string,\
    classifier_to_string, regressor_to_string


def data_file_create(path, X, Y):
    X_Y = np.column_stack((X, Y[np.newaxis].T))
    np.savetxt(path, X_Y)


def data_storlet_file_open(path):
    fd = os.open(path, os.O_RDONLY)
    sif = files.StorletInputFile(dict(), fd)
    return sif


def data_file_read(path, num_features, num_labels):
    sif = data_storlet_file_open(path)
    loadedX_Y = np.loadtxt(sif)
    sif.close()
    num_colums = num_features + num_labels
    num_samples = loadedX_Y.size / num_colums
    loadedX_Y = np.reshape(loadedX_Y, (num_samples, num_colums))
    X, Y, junk = np.hsplit(loadedX_Y, np.array((num_features, num_colums)))
    return X, Y.ravel()


def data_file_destroy(path):
    os.unlink(path)


def estimator_from_string(est_type, sest):
    if est_type == 'SGDRegressor':
        return regressor_from_string(sest)
    if est_type == 'SGDClassifier':
        return classifier_from_string(sest)


def estimator_to_string(est_type, est):
    if est_type == 'SGDRegressor':
        return regressor_to_string(est)
    if est_type == 'SGDClassifier':
        return classifier_to_string(est)


def get_estimator(est_type):
    if est_type == 'SGDRegressor':
        return SGDRegressor(shuffle=False)
    if est_type == 'SGDClassifier':
        return SGDClassifier(shuffle=False)
