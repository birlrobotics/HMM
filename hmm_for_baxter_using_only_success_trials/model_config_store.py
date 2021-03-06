model_store = {
    'hmmlearn\'s HMM': {
        'use': 'c1_less_iter_less_maxhstate',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'diag',
            },
            'c1': {
                'hmm_max_train_iteration': 100000,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            'c1_less_iter': {
                'hmm_max_train_iteration': 1000,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            'c1_less_iter_less_maxhstate': {
                'hmm_max_train_iteration': 1000,
                'hmm_max_hidden_state_amount': 10,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            'a1': {
                'hmm_max_train_iteration': [100, 1000],
                'hmm_hidden_state_amount': [1,2,3,4,5,6,7],
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
            '201709051536': {
                'hmm_max_train_iteration': 100000,
                'hmm_hidden_state_amount': 5,
                'gaussianhmm_covariance_type_string': 'full',
            },
            'config_that_make_state_1_diverge_for_20170711data': {
                'hmm_max_train_iteration': 100000,
                'hmm_hidden_state_amount': 2,
                'gaussianhmm_covariance_type_string': 'diag',
            },
        }
    },
    'BNPY\'s HMM': {
        'use': 'b1',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'alloModel' : 'HDPHMM',     
                'obsModel'  : 'Gauss',     
                'varMethod' : 'moVB',
            },
            'b1': {
                'hmm_max_train_iteration': 100000,
                'hmm_max_hidden_state_amount': 100,
                'alloModel' : 'HDPHMM',     
                'obsModel'  : ['Gauss', 'DiagGauss', 'ZeroMeanGauss'],     
                'varMethod' : ['moVB', 'memoVB'],
            },
        }
    },
    'hmmlearn\'s GMMHMM': {
        'use': 'd1',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'diag',
                'GMM_state_amount': 10,
            },
            'haha': {
                'hmm_max_train_iteration': 100000,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'full',
                'GMM_state_amount': 10,
            },
            'c1': {
                'hmm_max_train_iteration': 100000,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
                'GMM_state_amount': [1,2,3,4,5,6,7,8,9,10],
            },
            'd1': {
                'hmm_max_train_iteration': 100,
                'hmm_max_hidden_state_amount': 100,
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
                'GMM_max_state_amount': 100,
            },
        }
    },
}
