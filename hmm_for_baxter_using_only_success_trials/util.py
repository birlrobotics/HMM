import numpy as np
import ipdb

def convert_camel_to_underscore(name):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_model_config_id(model_config):
    model_id = ''
    for config_key in model_config:
        uncamel_key = convert_camel_to_underscore(config_key)
        for word in uncamel_key.split('_'): 
            model_id += word[0]
        model_id += '_(%s)_'%(model_config[config_key],)
    return model_id

def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

def fast_log_curve_calculation(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        logprobij, _fwdlattice = model._do_forward_pass(framelogprob)

        log_curve = [logsumexp(_fwdlattice[i]) for i in range(len(_fwdlattice))]

        return log_curve 
    elif issubclass(type(model.model), bnpy.HModel):
        return model.calc_log(X)
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))


def get_hidden_state_log_prob_matrix(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        logprobij, _fwdlattice = model._do_forward_pass(framelogprob)

        return _fwdlattice 
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

def get_emission_log_prob_matrix(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])

        return framelogprob 
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))


def make_trials_of_each_state_the_same_length(_trials_group_by_folder_name):
    import copy

    # may implement DTW in the future...
    # for now we just align trials with the shortest trial of each state

    trials_group_by_folder_name = copy.deepcopy(_trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    for state_no in range(1, state_amount+1):

        min_length = None
        for trial_name in trials_group_by_folder_name:
            # remember that the actual data is a numpy matrix
            # so we use *.shape[0] to get the length
            now_length = trials_group_by_folder_name[trial_name][state_no].shape[0]
            if min_length is None or now_length < min_length:
                min_length = now_length

        # align all trials in this state to min_length
        for trial_name in trials_group_by_folder_name:
            trials_group_by_folder_name[trial_name][state_no] = trials_group_by_folder_name[trial_name][state_no][:min_length, :]

    return trials_group_by_folder_name







