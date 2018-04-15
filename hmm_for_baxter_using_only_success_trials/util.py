import numpy as np
import ipdb
import os

def convert_camel_to_underscore(name):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_config_name_abbr(config_name):
    abbr = ''
    uncamel_key = convert_camel_to_underscore(config_name)
    for word in uncamel_key.split('_'): 
        abbr += word[0]
    return abbr

def get_model_config_id(model_config):
    model_id = ''
    for config_key in model_config:
        model_id += '%s_(%s)_'%(get_config_name_abbr(config_key), model_config[config_key])
    return model_id

def parse_model_config_id(model_id):
    items = model_id.strip('_').split('_')
    model_config = {}
    for idx in range(0, len(items), 2):
        model_config[items[idx]] = items[idx+1][1:-1]
        
    return model_config

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


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        a_log = np.log(a)
        a_log[a <= 0] = 0.0
        return a_log


def get_log_transmat(model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        log_transmat = log_mask_zero(model.transmat_)

        return log_transmat 
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

def get_trials_group_by_folder_name(training_config, data_class='success'):
    import copy

    if data_class == 'success':
        data_path = training_config.success_path
    elif data_class == 'anomaly':
        data_path = training_config.anomaly_data_path
    elif data_class == 'test_success':
        data_path = training_config.test_success_data_path
    else:
        raise Exception("unknown data class %s"%data_class)

    import load_csv_data
    trials_group_by_folder_name, state_order_group_by_folder_name = load_csv_data.run(
        data_path = data_path,
        interested_data_fields = training_config.interested_data_fields,
        preprocessing_normalize = training_config.preprocessing_normalize,
        preprocessing_scaling = training_config.preprocessing_scaling
    )

    return trials_group_by_folder_name, state_order_group_by_folder_name

def inform_config(training_config):
    import json
    config_to_print = [
        'training_config.config_by_user',
        'training_config.interested_data_fields',
        'training_config.model_config',
        'training_config.model_id',
    ]
    
    for s in config_to_print:
        print '-'*20
        print s, ':'
        print json.dumps(
            eval(s),
            indent=4,
        )
    print '#'*20
    print "press any key to continue."
    raw_input()

def bring_model_id_back_to_model_config(model_id, template):
    import copy
    config_to_return = copy.deepcopy(template)
    str_model_config = parse_model_config_id(model_id)
    for config_key in config_to_return:
        type_of_value = type(config_to_return[config_key])
        config_to_return[config_key] = type_of_value(str_model_config[get_config_name_abbr(config_key)])

    return config_to_return 

def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        a_log = np.log(a)
        a_log[a <= 0] = 0.0
        return a_log


def fast_viterbi_lock_t_cal(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        n_samples, n_components = framelogprob.shape
        log_startprob = log_mask_zero(model.startprob_)
        log_transmat = log_mask_zero(model.transmat_)
        work_buffer = np.empty(n_components)

        list_of_lock_t = []


        viterbi_lattice = np.zeros((n_samples, n_components))
        viterbi_trace = np.zeros((n_samples, n_components))
        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]
            viterbi_trace[0, i] = 0

        list_of_lock_t.append(None)

        # Induction
        for t in range(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])

                prev_state = np.argmax(work_buffer)
                viterbi_lattice[t, i] = work_buffer[prev_state] + framelogprob[t, i]
                viterbi_trace[t, i] = prev_state

            # backtract 
            lock_t = None
            for k in range(t, 0, -1):
                if np.all(viterbi_trace[k, :] == viterbi_trace[k, 0]):
                    lock_t = k-1
                    break
            

            list_of_lock_t.append(lock_t)
            

        return list_of_lock_t, n_samples, n_components
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))



def fast_growing_viterbi_paths_cal(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        n_samples, n_components = framelogprob.shape
        log_startprob = log_mask_zero(model.startprob_)
        log_transmat = log_mask_zero(model.transmat_)
        work_buffer = np.empty(n_components)

        list_of_growing_viterbi_paths = []


        viterbi_lattice = np.zeros((n_samples, n_components))
        viterbi_trace = np.zeros((n_samples, n_components))
        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]
            viterbi_trace[0, i] = 0

        list_of_growing_viterbi_paths.append([np.argmax(viterbi_lattice[0])])
        # Induction
        for t in range(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])

                prev_state = np.argmax(work_buffer)
                viterbi_lattice[t, i] = work_buffer[prev_state] + framelogprob[t, i]
                viterbi_trace[t, i] = prev_state

            best_state_at_t = np.argmax(viterbi_lattice[t, :])

            viterbi_path = [0 for k in range(t+1)]
            viterbi_path[t] = best_state_at_t
            # backtract 

            for k in range(t, 0, -1):
                forward_z = viterbi_path[k]
                viterbi_path[k-1] = int(viterbi_trace[k, forward_z])

            list_of_growing_viterbi_paths.append(viterbi_path)
            

        return list_of_growing_viterbi_paths, n_samples, n_components
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))


def rgba_to_rgb_using_white_bg(rgb_array, alpha):
    return [i*alpha+(1-alpha) for i in rgb_array]
    

def gray_a_pixel(pixel):
    import numpy as np
    p = np.array(pixel)
    p += 50
    p /= 2
    return (p[0], p[1], p[2])

def output_growing_viterbi_path_img(
    list_of_growing_viterbi_paths, 
    hidden_state_amount, 
    output_file_path,
    list_of_lock_t=None,
):
    from matplotlib.pyplot import cm
    import numpy as np

    height = len(list_of_growing_viterbi_paths)
    width = len(list_of_growing_viterbi_paths[-1])

    colors = [tuple((256*i).astype(int)) for i in cm.rainbow(np.linspace(0, 1, hidden_state_amount))]

    output_pixels = []

    for idx, vp in enumerate(list_of_growing_viterbi_paths):
        black_to_append = width-len(vp)
        row = [colors[i] for i in vp]+[(0,0,0) for i in range(black_to_append)]
        if list_of_lock_t is not None:
            lock_t = list_of_lock_t[idx]
            if lock_t is not None:
                for i in range(lock_t+1):
                    row[i] = gray_a_pixel(row[i])

        output_pixels += row

    from PIL import Image

    output_dir = os.path.dirname(output_file_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_img = Image.new("RGB", (width, height)) # mode,(width,height)
    output_img.putdata(output_pixels)
    output_img.save(
        output_file_path,
    )

def _norm_loglik(a):
    from scipy.misc import logsumexp
    s = logsumexp(a)
    a = np.exp(a-s)
    return a

def visualize_viterbi_alog(X, model, path):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nodes_x = []
    nodes_y = []
    nodes_color = []
    edges = []

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        n_samples, n_components = framelogprob.shape
        log_startprob = log_mask_zero(model.startprob_)
        log_transmat = log_mask_zero(model.transmat_)
        work_buffer = np.empty(n_components)
        connection = np.empty((n_components, n_components))

        list_of_growing_viterbi_paths = []


        viterbi_lattice = np.zeros((n_samples, n_components))
        viterbi_trace = np.zeros((n_samples, n_components))
        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]
            viterbi_trace[0, i] = 0
        tmp = _norm_loglik(viterbi_lattice[0])
        for i in range(n_components):
            nodes_x.append(0)
            nodes_y.append(i)
            nodes_color.append('#%02x%02x%02x%02x'%(0, 0, 0, int(255*tmp[i])))

        # Induction
        for t in range(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])
                    connection[j, i] = log_transmat[j, i]+framelogprob[t, i]

                prev_state = np.argmax(work_buffer)
                viterbi_lattice[t, i] = work_buffer[prev_state] + framelogprob[t, i]
                viterbi_trace[t, i] = prev_state

            tmp = _norm_loglik(connection.flatten()).reshape((n_components, n_components))
            for i in range(n_components):
                for j in range(n_components):
                    edges.append((t-1,t))
                    edges.append((j,i))
                    edges.append('#%02x%02x%02x%02x'%(0, 0, 0, int(255*tmp[j][i])))
                edges.append((t-1,t))
                edges.append((viterbi_trace[t, i]+0.1,i+0.1))
                edges.append('k:')

            tmp = _norm_loglik(viterbi_lattice[t])
            for i in range(n_components):
                nodes_x.append(t)
                nodes_y.append(i)
                nodes_color.append('#%02x%02x%02x%02x'%(0, 0, 0, int(255*tmp[i])))

            best_state_at_t = np.argmax(viterbi_lattice[t, :])
            nodes_x.append(t)
            nodes_y.append(best_state_at_t+0.1)
            nodes_color.append('r')
    else:
        raise Exception('model of type %s is not supported by visualize_viterbi_alog.'%(type(model),))
    
    ax.plot(*edges)
    ax.scatter(x=nodes_x, y=nodes_y, c=nodes_color)
    fig.set_size_inches(2*n_samples, 2)
    fig.savefig(path)
    print 'done one viterbi alog graph'
