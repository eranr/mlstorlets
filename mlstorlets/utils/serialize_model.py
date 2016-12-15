import json
import numpy
import StringIO
from sklearn.linear_model import SGDRegressor, SGDClassifier

def serialize_narray(a):
    memfile = StringIO.StringIO()
    numpy.save(memfile, a)
    memfile.seek(0)
    return json.dumps(memfile.read().decode('latin-1'))

def deserialize_narray(sa):
    memfile = StringIO.StringIO()
    memfile.write(json.loads(sa).encode('latin-1'))
    memfile.seek(0)
    return numpy.load(memfile)

def estimator_to_string(est):
    params = est.get_params(deep=True)
    # coef_ can be either None or array
    params['coef_'] = None if est.coef_ is None else serialize_narray(est.coef_)
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
            params['average_intercept_'] = serialize_dnarray(est.average_intercept_)
            params['standard_intercept_'] = serialize_dnarray(est.standard_intercept_)
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
    est.coef_ = None if params['coef_'] is None else deserialize_narray(params['coef_'])

    try:
        # intercept either exists or doesn't
        est.intercept_ = est.standard_intercept_ =\
            deserialize_narray(params['intercept_'])
    except Exception:
        pass

    if params['average'] > 0:
        # average_coef_ can be either None or array
        est.average_coef_ = None if params['average_coef_'] is None else deserialize_narray(params['average_coef_'])
        est.standard_coef_ = None if params['standard_coef_'] is None else deserialize_narray(params['standard_coef_'])

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
