# Copyright (c) 2010-2016 OpenStack Foundation
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
import ConfigParser
from swiftclient import client


def parse_config(config_file):
    required_keys = ['auth_version',
                     'user',
                     'password',
                     'project_name',
                     'domain_name',
                     'auth_uri']
    conf = dict()
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    options = config.options('Default')
    for option in options:
        try:
            conf[option] = config.get('Default', option)
        except:
            raise

    if all([k in conf for k in required_keys]) is False:
        raise Exception('Missing access information')

    return conf


def get_auth(conf):
    """
    Get token string to access to swift

    :param conf: a dict of config parameters
    :returns: (swift endpoint url, token string)
    """
    auth_url = conf['auth_uri']
    project = conf['project_name']
    os_options = {'user_domain_name': conf['domain_name'],
                  'project_name': conf['project_name']}
    user = conf['user']
    passwd = conf['password']
    url, token = client.get_auth(auth_url, project + ':' + user, passwd,
                                 os_options=os_options,
                                 auth_version=conf['auth_version'])
    return url, token


def put_local_file(url, token, container, local_dir, local_file, headers=None):
    """
    Put local file to swift

    :param url: swift endpoint url
    :param token: token string to access to swift
    :param local_dir: directory path where the target file is placed
    :param loca_file: name of the file to be put to swift
    :param headers: headers parameters to be included in request headers
    """
    resp = dict()
    with open(os.path.join(local_dir, local_file), 'r') as f:
        client.put_object(url, token, container, local_file, f,
                          headers=headers,
                          content_type="application/octet-stream",
                          response_dict=resp)
        status = resp.get('status', 0)
        assert (status // 100 == 2)


def deploy_mlstorlet(conf, path_to_storlet):
    url, token = get_auth(conf)

    headers = {'X-Object-Meta-Storlet-Language': 'Python',
               'X-Object-Meta-Storlet-Interface-Version': '1.0',
               'X-Object-Meta-Storlet-Object-Metadata': 'no',
               'X-Object-Meta-Storlet-Main': 'linear_model.SGDEstimator'}
    put_local_file(url, token, 'storlet', os.path.dirname(path_to_storlet),
                   os.path.basename(path_to_storlet), headers)


def _parse_data_url(data_url):
    url_elements = data_url.split('/')
    url_elements = [el for el in url_elements if el != '']
    return url_elements[0], url_elements[1]


def invoke_storlet(url, token,
                   data_url,
                   est_type, command, sest,
                   num_features=None, num_labels=None,
                   sample_weight=None,
                   coef_init=None, intercept_init=None,
                   proxy_invoke=False):
    container_name, object_name = _parse_data_url(data_url)

    params = {'type': est_type,
              'command': command}
    if num_features:
        params['num_features'] = num_features
    if num_labels:
        params['num_labels'] = num_labels
    if coef_init:
        params['coef_init'] = coef_init
    if intercept_init:
        params['intercept_init'] = intercept_init
    params['serialized_estimator'] = sest
    headers = dict()
    i = 1
    for key in params:
        if params[key]:
            headers['X-Storlet-Parameter-%d' % i] =\
                '%s:%s' % (key, params[key])
            i = i + 1
    headers['X-Run-Storlet'] = 'linear_model.py'
    if proxy_invoke is True:
        headers['X-Storlet-Run-On-Proxy'] = ''

    rest_headers, resp_content = client.get_object(
        url, token,
        container_name, object_name,
        response_dict={},
        headers=headers)

    return resp_content
