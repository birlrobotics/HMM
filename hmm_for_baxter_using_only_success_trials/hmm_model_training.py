#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)
import util
    
now_score = None

def get_model_generator(model_type, model_config):
    global now_score

    if model_type == 'hmmlearn\'s HMM':
        import hmmlearn.hmm 
        if type(model_config['hmm_max_train_iteration']) is not list:
            model_config['hmm_max_train_iteration'] = [model_config['hmm_max_train_iteration']]

        if type(model_config['gaussianhmm_covariance_type_string']) is not list:
            model_config['gaussianhmm_covariance_type_string'] = [model_config['gaussianhmm_covariance_type_string']]

        if 'hmm_max_hidden_state_amount' in model_config:
            model_config['hmm_hidden_state_amount'] = range(1, model_config['hmm_max_hidden_state_amount']+1)
        else:
            if type(model_config['hmm_hidden_state_amount']) is not list:
                model_config['hmm_hidden_state_amount'] = [model_config['hmm_hidden_state_amount']]


        for covariance_type in model_config['gaussianhmm_covariance_type_string']:
            for n_iter in model_config['hmm_max_train_iteration']:
                now_score = None
                last_score = None
            
                for n_components in model_config['hmm_hidden_state_amount']:
                    model = hmmlearn.hmm.GaussianHMM(
                        n_components=n_components, 
                        covariance_type=covariance_type,
                        params="mct", 
                        init_params="cmt", 
                        n_iter=n_iter)
                    start_prob = np.zeros(n_components)
                    start_prob[0] = 1
                    model.startprob_ = start_prob

                    now_model_config = {
                        "hmm_hidden_state_amount": n_components,
                        "gaussianhmm_covariance_type_string": covariance_type,
                        "hmm_max_train_iteration": n_iter,
                    }

                    # we want a minimal score
                    if now_score is None or last_score is None:
                        pass
                    elif now_score < last_score:
                        # we are making good progress, don't stop
                        pass
                    else:
                        # seems best of the hidden state amount is hit, we'll stop if we've tried 1 to 5
                        if n_components > 5:
                            break
                        
                    last_score = now_score
                    yield model, now_model_config 

    elif model_type == 'BNPY\'s HMM':
        import hongminhmmpkg.hmm


        if type(model_config['hmm_max_train_iteration']) is not list:
            model_config['hmm_max_train_iteration'] = [model_config['hmm_max_train_iteration']]


        if type(model_config['alloModel']) is not list:
            model_config['alloModel'] = [model_config['alloModel']]

        if type(model_config['obsModel']) is not list:
            model_config['obsModel'] = [model_config['obsModel']]

        if type(model_config['varMethod']) is not list:
            model_config['varMethod'] = [model_config['varMethod']]

        if 'hmm_max_hidden_state_amount' in model_config:
            model_config['hmm_hidden_state_amount'] = range(1, model_config['hmm_max_hidden_state_amount']+1)
        else:
            if type(model_config['hmm_hidden_state_amount']) is not list:
                model_config['hmm_hidden_state_amount'] = [model_config['hmm_hidden_state_amount']]


        for alloModel in model_config['alloModel']:
            for obsModel in model_config['obsModel']:
                for varMethod in model_config['varMethod']:
                    for n_iter in model_config['hmm_max_train_iteration']:
                        now_score = None
                        last_score = None
                    
                        for n_components in model_config['hmm_hidden_state_amount']:

                            model = hongminhmmpkg.hmm.HongminHMM(
                                alloModel=alloModel,
                                obsModel=obsModel,
                                varMethod=varMethod,
                                n_iteration=n_iter,
                                K=n_components
                            )

                            now_model_config = {
                                'alloModel': alloModel,
                                'obsModel': obsModel,
                                'varMethod': varMethod,
                                'hmm_hidden_state_amount': n_components,
                                'hmm_max_train_iteration': n_iter,
                            }

                            # we want a minimal score
                            if now_score is None or last_score is None:
                                pass
                            elif now_score < last_score:
                                # we are making good progress, don't stop
                                pass
                            else:
                                # seems best of the hidden state amount is hit, we'll stop if we've tried 1 to 5
                                if n_components > 5:
                                    break
                                
                            last_score = now_score
                            yield model, now_model_config 

def run(model_save_path, 
    model_type,
    model_config,
    trials_group_by_folder_name):

    global now_score

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
        model_generator = get_model_generator(model_type, model_config)
        for model, now_model_config in model_generator:
            print 'in state', state_no, ' working on config:', now_model_config,


            X = training_data_group_by_state[state_no]
            lengths = training_length_array_group_by_state[state_no]

            model = model.fit(X, lengths=lengths)

            

            slice_10_time_step_log_lik = [[model.score(X[i:i+k*(j-i)/10]) for k in range(1, 11, 1)] for i, j in util.iter_from_X_lengths(X, lengths)]
            matrix = np.matrix(slice_10_time_step_log_lik)
            slice_10_means = abs(matrix.mean(0))
            slice_10_std = matrix.std(0)
            slice_10_stme_ratio = slice_10_std/slice_10_means
            std_mean_ratio = slice_10_stme_ratio.max()

            now_score = std_mean_ratio
        
            model_list.append({
                "model": model,
                "now_model_config": now_model_config,
                "std_mean_ratio": std_mean_ratio
            })
            print ' std_mean_ratio:', std_mean_ratio 

        sorted_model_list = sorted(model_list, key=lambda x:x['std_mean_ratio'])

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
                "model_s%s_std_mean_ratio_%s.pkl"%(state_no, best['std_mean_ratio'])
            )
        )

        train_report = [{util.get_model_config_id(i['now_model_config']): i['std_mean_ratio']} for i in sorted_model_list]
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


