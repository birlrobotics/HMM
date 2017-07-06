#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import hmmlearn.hmm
from sklearn.externals import joblib
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
import ipdb

def make_trials_of_each_state_the_same_length(trials_group_by_folder_name):
    # may implement DTW in the future...
    # for now we just align trials with the shortest trial of each state

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

def assess_threshold_and_decide(mean_of_log_curve, var_of_log_curve, np_matrix_traj_by_time, curve_owner, state_no, figure_save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot var value boundaries
    interested_boundary = [1, 10, 100, 1000]
    idx = 0
    for col_no in range(var_of_log_curve.shape[1]):
        if var_of_log_curve[0, col_no] > interested_boundary[idx]:
            ax.axvline(x = col_no) 
            ax.text(col_no, 0, 'var>%s'%(interested_boundary[idx],), rotation=90)
            idx += 1
            if idx == len(interested_boundary):
                break;
        

    # plot log curves of all trials
    for row_no in range(np_matrix_traj_by_time.shape[0]):
        trial_name = curve_owner[row_no]
        ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="solid", color='gray')

    # plot mean log curve
    ax.plot(mean_of_log_curve.tolist()[0], label='mean', linestyle="solid")

    # plot mean-c*var log curve
    for c in np.arange(0.1, 1.1, 0.1):
        ax.plot((mean_of_log_curve-c*var_of_log_curve).tolist()[0], label="mean-%s*var"%(c,), linestyle='solid')

    ax.legend()

    fig.show()

    # decide c in an interactive way
    print 
    print 
    print 
    print "enter c (default 0.1) to visualize mean-c*var or enter ok to use this c as final threshold:"
    c = 0.1 # this is default
    while True:
        i_str = raw_input()
        if i_str == 'ok':
            title = 'state %s use threshold with c=%s'%(state_no, c)
            ax.set_title(title)
            if not os.path.isdir(figure_save_path+'/threshold_assessment'):
                os.makedirs(figure_save_path+'/threshold_assessment')
            fig.savefig(os.path.join(figure_save_path, 'threshold_assessment', title+".eps"), format="eps")
            return mean_of_log_curve-c*var_of_log_curve
        try:
            c = float(i_str)
            ax.plot((mean_of_log_curve-c*var_of_log_curve).tolist()[0], label="mean-%s*var"%(c,), linestyle='dotted')
            ax.legend()
            fig.show()
        except ValueError:
            print 'bad input'

            
        
        
    
def run(model_save_path, 
    figure_save_path,
    trials_group_by_folder_name):


        
    trials_group_by_folder_name = make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    threshold_constant = 10
    threshold_offset = 10

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        model_group_by_state[state_no] = joblib.load(model_save_path+"/multisequence_model/model_s%s.pkl"%(state_no,))

    expected_log = []
    var_of_log = []
    threshold = []

    for state_no in range(1, state_amount+1):
        all_log_curves_of_this_state = []
        curve_owner = []
        for trial_name in trials_group_by_folder_name:
            curve_owner.append(trial_name)
            one_log_curve_of_this_state = [] 
            for time_step in range(len(trials_group_by_folder_name[trial_name][state_no])):
                log_probability = model_group_by_state[state_no].score(trials_group_by_folder_name[trial_name][state_no][:time_step+1])
                one_log_curve_of_this_state.append(log_probability)
            all_log_curves_of_this_state.append(one_log_curve_of_this_state)

        # use np matrix to facilitate the computation of mean curve and var 
        np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)
        mean_of_log_curve = np_matrix_traj_by_time.mean(0)
        var_of_log_curve = np_matrix_traj_by_time.var(0)

        decided_threshold_log_curve = assess_threshold_and_decide(mean_of_log_curve, var_of_log_curve, np_matrix_traj_by_time, curve_owner, state_no, figure_save_path)
        expected_log.append(mean_of_log_curve.tolist()[0])
        threshold.append(decided_threshold_log_curve.tolist()[0])
        var_of_log.append(var_of_log_curve.tolist()[0])

    if not os.path.isdir(model_save_path+"/multisequence_model"):
        os.makedirs(model_save_path+"/multisequence_model")
        
    joblib.dump(expected_log, model_save_path+"/multisequence_model/expected_log.pkl")
    joblib.dump(threshold, model_save_path+"/multisequence_model/threshold.pkl")
    joblib.dump(var_of_log, model_save_path+"/multisequence_model/var_of_log.pkl")
