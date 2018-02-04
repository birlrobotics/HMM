#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)
import util
import ipdb
import copy
from birl_hmm.hmm_training import train_model
    
def run(model_save_path, 
    model_type,
    model_config,
    score_metric,
    trials_group_by_folder_name,
    test_trials_group_by_folder_name
):


    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)
    list_of_training_trial = trials_group_by_folder_name.values() 


    test_trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(test_trials_group_by_folder_name)
    list_of_test_trial = test_trials_group_by_folder_name.values() 

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    one_trial_data_group_by_state = list_of_training_trial[0]
    state_amount = len(one_trial_data_group_by_state)




    training_data_group_by_state = {}
    test_data_group_by_state = {}
    for state_no in range(1, state_amount+1):
        training_data_group_by_state[state_no] = []
        test_data_group_by_state[state_no] = []
        for trial_no in range(len(list_of_training_trial)):
            training_data_group_by_state[state_no].append( 
                list_of_training_trial[trial_no][state_no]
            )
        for trial_no in range(len(list_of_test_trial)):
            test_data_group_by_state[state_no].append( 
                list_of_test_trial[trial_no][state_no]
            )

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    for state_no in range(1, state_amount+1):
        print 'state_no', state_no
        sorted_model_list = train_model.run(
            list_of_train_mat = training_data_group_by_state[state_no],
            list_of_test_mat = test_data_group_by_state[state_no],
            model_type=model_type,
            model_config=model_config,
            score_metric=score_metric,
        )

        best = sorted_model_list[0]
        model_id = util.get_model_config_id(best['now_model_config'])

        joblib.dump(
            best['model'],
            os.path.join(model_save_path, "model_s%s.pkl"%(state_no,))
        )
    
        joblib.dump(
            best['now_model_config'], 
            os.path.join(
                model_save_path, 
                "model_s%s_config_%s.pkl"%(state_no, model_id)
            )
        )

        joblib.dump(
            None,
            os.path.join(
                model_save_path, 
                "model_s%s_score_%s.pkl"%(state_no, best['score'])
            )
        )

        train_report = [{util.get_model_config_id(i['now_model_config']): i['score']} for i in sorted_model_list]
        import json
        json.dump(
            train_report, 
            open(
                os.path.join(
                    model_save_path, 
                    "model_s%s_training_report.json"%(state_no)
                ), 'w'
            ),
            separators = (',\n', ': ')
        )


