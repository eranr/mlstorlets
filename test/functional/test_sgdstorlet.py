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
import unittest
import uuid

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import SGDClassifier, SGDRegressor
from swiftclient import client
from mlstorlets.utils.swift_access import parse_config,\
   get_auth, put_local_file, deploy_mlstorlet,\
   invoke_storlet 
from mlstorlets.utils.serialize_model import\
    classifier_from_string, classifier_to_string,\
    regressor_from_string, regressor_to_string,\
    deserialize_narray

from test import data_file_create, data_file_destroy,\
    data_file_read 

class TestSGDStorlet(unittest.TestCase):
    def setUp(self):
        self.container_name=str(uuid.uuid4())
        conf = parse_config('access.cfg')
        self.url, self.token = get_auth(conf)

        deploy_mlstorlet(conf, 'mlstorlets/sgdstorlet/linear_model.py')

        client.put_container(self.url, self.token, self.container_name)

        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=1, cluster_std=0.60)
        data_file_create('/tmp/data1', X, Y)
        put_local_file(self.url, self.token, self.container_name,
                       '/tmp','data1')

        X, Y = make_blobs(n_samples=1000, centers=2,
                          random_state=1, cluster_std=0.60)
        data_file_create('/tmp/data2', X, Y)
        headers={'X-Object-Meta-num-features': '2',
                 'X-Object-Meta-num-labels': '1'}
        put_local_file(self.url, self.token, self.container_name,
                       '/tmp','data2', headers)

    def tearDown(self):
        client.delete_object(self.url, self.token, self.container_name, 'data1')
        client.delete_object(self.url, self.token, self.container_name, 'data2')
        client.delete_container(self.url, self.token, self.container_name)
        data_file_destroy('/tmp/data1')
        data_file_destroy('/tmp/data2')

    def _test_invoke_fit_shape_from_param(self, proxy_invoke):
        data_url='%s/%s' % (self.container_name, 'data1')
        Xtest = [[0.7, 0.5], [2, 1]]
        reg = SGDRegressor(shuffle=False)
        sreg = regressor_to_string(reg) 
        res = invoke_storlet(self.url, self.token,
                             data_url, 'SGDRegressor', 'fit',
                             sreg, '2', '1', proxy_invoke)
        reg = regressor_from_string(res)
        p1 = reg.predict(Xtest)

        reg = SGDRegressor(shuffle=False)
        X, Y = data_file_read('/tmp/data1', 2, 1)
        reg.fit(X,Y)
        p2 = reg.predict(Xtest)

        self.assertTrue(np.array_equal(p1, p2))

    def _test_invoke_fit_shape_from_metadata(self, proxy_invoke):
        data_url='%s/%s' % (self.container_name, 'data2')
        Xtest = [[0.7, 0.5], [2, 1]]
        reg = SGDRegressor(shuffle=False)
        sreg = regressor_to_string(reg) 
        res = invoke_storlet(self.url, self.token,
                             data_url, 'SGDRegressor', 'fit',
                             sreg, proxy_invoke=proxy_invoke)
        reg = regressor_from_string(res)
        p1 = reg.predict(Xtest)

        reg = SGDRegressor(shuffle=False)
        X, Y = data_file_read('/tmp/data2', 2, 1)
        reg.fit(X,Y)
        p2 = reg.predict(Xtest)

        self.assertTrue(np.array_equal(p1, p2))

    def test_fit_params_object_node(self):
        self._test_invoke_fit_shape_from_param(False)

    def test_fit_params_proxy_node(self):
        self._test_invoke_fit_shape_from_param(True)

    def test_fit_metadata_object_node(self):
        self._test_invoke_fit_shape_from_metadata(False)

    def test_fit_metadata_proxy_node(self):
        self._test_invoke_fit_shape_from_metadata(True)
