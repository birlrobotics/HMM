model_store = {
    'hmmlearn\'s HMM': {
        'use': 'default',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'diag',
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
