model_store = {
    'hmmlearn\'s HMM': {
        'use': 'crazy',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'diag',
            },
            'crazy': {
                'hmm_max_train_iteration': [10, 100, 1000],
                'hmm_hidden_state_amount': [4, 8, 16, 32],
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
        }
    }
}
