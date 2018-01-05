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
import model_generation
import model_score
    
def run(model_save_path, 
    model_type,
    model_config,
    score_metric,
        trials_group_by_folder_name, **kargs):

    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)
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

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    for state_no in range(1, state_amount+1):
        model_list = []
        model_generator = model_generation.get_model_generator(model_type, model_config)
        for model, now_model_config in model_generator:
            print
            print '-'*20
            print 'in state', state_no, ' working on config:', now_model_config

            X = training_data_group_by_state[state_no]
            lengths = training_length_array_group_by_state[state_no]
            model = model.fit(X, lengths=lengths, state_no = state_no, **kargs)

            score = model_score.score(score_metric, model, X, lengths)
            if score == None:
                print "scorer says to skip this model, will do"
                continue

            model_list.append({
                "model": model,
                "now_model_config": now_model_config,
                "score": score
            })
            print 'score:', score 
            print '='*20
            print 

            model_generation.update_now_score(score)

        sorted_model_list = sorted(model_list, key=lambda x:x['score'])

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


