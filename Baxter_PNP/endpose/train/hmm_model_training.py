#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)
    
def get_model_generator(model_type, model_config):
    if model_type == 'hmmlearn\'s HMM':
        def model_generator():
            import hmmlearn.hmm 
            model = hmmlearn.hmm.GaussianHMM(
                n_components=model_config['hmm_hidden_state_amount'], 
                covariance_type=model_config['gaussianhmm_covariance_type_string'],
                params="mct", 
                init_params="cmt", 
                n_iter=model_config['hmm_max_train_iteration'])
            start_prob = np.zeros(model_config['hmm_hidden_state_amount'])
            start_prob[0] = 1
            model.startprob_ = start_prob
            return model
    elif model_type == 'BNPY\'s HMM':
        pass

    return model_generator 

def run(model_save_path, 
    model_type,
    model_config,
    trials_group_by_folder_name):

    list_of_trials = trials_group_by_folder_name.values() 

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    one_trial_data_group_by_state = list_of_trials[0]
    state_amount = len(one_trial_data_group_by_state)




    training_data_group_by_state = {}
    training_length_array_group_by_state = {}

    for state_no in range(1, state_amount+1):
        length_array = []
        for trial_no in range(len(list_of_trials)):
            length_array.append(list_of_trials[trial_no][state_no].shape[0])
            if trial_no == 0:
                data_tempt = list_of_trials[trial_no][state_no]
            else:
                data_tempt = np.concatenate((data_tempt,list_of_trials[trial_no][state_no]),axis = 0)
        training_data_group_by_state[state_no] = data_tempt
        training_length_array_group_by_state[state_no] = length_array


    model_generator = get_model_generator(model_type, model_config)
    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        model = model_generator()
        model = model.fit(
            training_data_group_by_state[state_no],
            lengths=training_length_array_group_by_state[state_no])
    
        model_group_by_state[state_no] = model


    # save the models
    if not os.path.isdir(model_save_path+"/multisequence_model"):
        os.makedirs(model_save_path+"/multisequence_model")
    for state_no in range(1, state_amount+1):
        joblib.dump(
            model_group_by_state[state_no], 
            model_save_path+"/multisequence_model/model_s%s.pkl"%(state_no,))
    
    joblib.dump(model_config, model_save_path+"/multisequence_model/model_config.pkl")
