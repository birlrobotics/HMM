model_store = {
    'hmmlearn\'s HMM': {
        'use': 'c1',
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
            'a1': {
                'hmm_max_train_iteration': [100, 1000],
                'hmm_hidden_state_amount': [1,2,3,4,5,6,7],
                'gaussianhmm_covariance_type_string': ['diag', 'spherical', 'full', 'tied'],
            },
        }
    },
    'BNPY\'s HMM': {
        'use': 'default',
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
        'use': 'haha',
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
        }
    },
}
