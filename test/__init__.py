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
import StringIO
import numpy as np
from storlets.agent.daemon import files

def data_file_create(path, X,Y):
    X_Y = np.column_stack((X, Y[np.newaxis].T))
    np.savetxt(path,X_Y)

def data_storlet_file_open(path):
    fd = os.open(path,os.O_RDONLY)
    sif = files.StorletInputFile(dict(), fd)
    return sif

def data_file_read(path, num_features, num_labels):
    sif = data_storlet_file_open(path)
    loadedX_Y = np.loadtxt(sif)
    sif.close()
    num_colums = num_features + num_labels
    num_samples = loadedX_Y.size / num_colums
    loadedX_Y = np.reshape(loadedX_Y,(num_samples, num_colums))
    cols=np.shape(loadedX_Y)[1]
    X, Y, junk=np.hsplit(loadedX_Y,np.array((num_features,num_colums)))
    return X,Y.ravel()

def data_file_destroy(path):
    os.unlink(path)
