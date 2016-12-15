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

from mlstorlets.utils.serialize_model import classifier_to_string, classifier_from_string
from swiftclient import client as swiftclient

def get_auth(conf_path):
    pass

def invoke(path, esttype, est, command):
    if esttype=='SGDRegressor':
        sest = regressor_to_string(est)
    elif esttype=='SGDClasifier':
        sest = classifier_to_string(est)
    else:
        raise Exception('Unkown estimator type')

    headers = {
        'X-Run-Storlet': '',
        'X-Storlet-Parameter-type': esttype,
        'X-Storlet-Parameter-serialised_estimator': sest,
        'X-Storlet-Parameter-command': command
    }
