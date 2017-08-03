#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
import time
import util

import ipdb



def plot_logpmf_of_all_trials(
    list_of_logpmf_mat,
    logpmf_owner, 
    state_no, 
    figure_save_path):


    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    from matplotlib.pyplot import cm 
    import numpy as np

    for i in range(len(list_of_logpmf_mat)):
        logpmf_mat = list_of_logpmf_mat[i].transpose()
        for col_no in range(logpmf_mat.shape[1]):
            max_exponent = max(logpmf_mat[:, col_no])
            logpmf_mat[:, col_no] -= max_exponent
            unnormalized_pmf = np.exp(logpmf_mat[:, col_no])
            psum = sum(unnormalized_pmf)
            normalized_pmf = unnormalized_pmf/psum
            logpmf_mat[:, col_no] = normalized_pmf

        hidden_state_amount = logpmf_mat.shape[0]
        color=iter(cm.rainbow(np.linspace(0, 1, hidden_state_amount)))
        for row_no in range(hidden_state_amount):
            c=next(color)

            if i == 0:
                ax.plot(logpmf_mat[row_no].tolist(), linestyle="solid", color=c, label='hidden state %s'%(row_no,))
            else:
                ax.plot(logpmf_mat[row_no].tolist(), linestyle="solid", color=c)

    title = 'state %s trial hidden state logpmf plot'%(state_no,)
    ax.set_title(title)
    ax.legend()

    plt.show()

    if not os.path.isdir(figure_save_path+'/hidden_state_logpmf_plot'):
        os.makedirs(figure_save_path+'/hidden_state_logpmf_plot')
    fig.savefig(os.path.join(figure_save_path, 'hidden_state_logpmf_plot', title+".eps"), format="eps")
    plt.close(1)
    
def run(model_save_path, 
    figure_save_path,
    threshold_c_value,
    trials_group_by_folder_name):


        
    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    threshold_constant = 10
    threshold_offset = 10

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        except IOError:
            print 'model of state %s not found'%(state_no,)
            continue

    expected_log = []
    std_of_log = []
    deri_threshold = []




    for state_no in model_group_by_state:

        list_of_logpmf_mat = []
        logpmf_owner = []
        for trial_name in trials_group_by_folder_name:
            logpmf_owner.append(trial_name)

            
            hidden_state_logpmf = util.get_hidden_state_logpmf_matrix(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )

            list_of_logpmf_mat.append(hidden_state_logpmf)

        # use np matrix to facilitate the computation of mean curve and std 
        plot_logpmf_of_all_trials(
            list_of_logpmf_mat, 
            logpmf_owner, 
            state_no, 
            figure_save_path)
