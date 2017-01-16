# mlstorlets
Machine Learning using Storlets

mlstorlets is an initial approach to leverage Storlets for
machine learning.

# TL;DR
On a fresh new 16.04 VM with a passwordless sudoer simply:

```
git clone https://github.com/eranr/mlstorlets.git
cd mlstorlets
./install.sh
tox -e functional
```

This will install Swift and Storlets on the VM together with
a docker container that has the scikit-learn package.

# Supported Algorithms
Currently, mlstorlets support Stochastic Grandiant Descent.
Specifically it implements SGDRegressorProxy,
SGDClassifierProxy which expose python's scikit-learn
SGDRegressor, SGDClassifier API [1] together with a
remote_fit and remote_score methods. Specifically,
the remote_fit method allow to do mini-batch SGD
where each batch comes either from local data and/or
one or more objects.

For more information see test/functional/test_sgdproxies.py

[1] http://scikit-learn.org/stable/modules/sgd.html
