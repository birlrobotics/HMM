#!/usr/bin/env python
import os
import numpy as np
import hmmlearn.hmm 
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)
    
def run(model_save_path, 
    n_state, 
    covariance_type_string, 
    n_iteration,
    trials_group_by_folder_name):

    list_of_trials = trials_group_by_folder_name.values() 

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    one_trial_data_group_by_state = list_of_trials[0]
    state_amount = len(one_trial_data_group_by_state)

    start_prob = np.zeros(state_amount)
    start_prob[0] = 1

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

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        model = hmmlearn.hmm.GaussianHMM(
            n_components=n_state, 
            covariance_type=covariance_type_string,
            params="mct", 
            init_params="cmt", 
            n_iter=n_iteration)
        model.startprob_ = start_prob
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
    
    joblib.dump(n_state, model_save_path+"/multisequence_model/n_state.pkl")
    joblib.dump(covariance_type_string, model_save_path+"/multisequence_model/covariance_type.pkl")

