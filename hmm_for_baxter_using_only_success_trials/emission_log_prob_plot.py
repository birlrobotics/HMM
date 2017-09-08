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
import math

import ipdb



def plot_log_prob_of_all_trials(
    list_of_log_prob_mat,
    log_prob_owner, 
    state_no, 
    figure_save_path
):


    trial_amount = len(list_of_log_prob_mat)
    hidden_state_amount = list_of_log_prob_mat[0].shape[1]


    subplot_per_row = 4

    subplot_amount = trial_amount*2
    row_amount = int(math.ceil(float(subplot_amount)/subplot_per_row))
    fig, ax_mat = plt.subplots(nrows=row_amount, ncols=subplot_per_row)
    if row_amount == 1:
        ax_mat = ax_mat.reshape(1, -1)


    ax_list = []
    for i in range(trial_amount):
        j = 2*i
        row_no = j/subplot_per_row
        col_no = j%subplot_per_row
        ax_list.append(ax_mat[row_no, col_no])

        j = 2*i+1
        row_no = j/subplot_per_row
        col_no = j%subplot_per_row
        ax_list.append(ax_mat[row_no, col_no])

    from matplotlib.pyplot import cm 
    import numpy as np

    colors_for_hstate = cm.rainbow(np.linspace(0, 1, hidden_state_amount))
    for trial_no in range(trial_amount):
        log_prob_mat = list_of_log_prob_mat[trial_no][:, :].transpose()

        plot_idx = 2*trial_no
        for hstate_no in range(hidden_state_amount):
            ax_list[plot_idx].plot(log_prob_mat[hstate_no].tolist(), linestyle="solid", color=colors_for_hstate[hstate_no])

        trial_name = log_prob_owner[trial_no]

        ax_list[plot_idx].set_title(trial_name)
        ymax = np.max(log_prob_mat)
        ax_list[plot_idx].set_ylim(ymin=0, ymax=ymax)


        vp_triangle_img = open(
            os.path.join(
                figure_save_path, 
                'check_if_viterbi_path_grow_incrementally',
                "state_%s"%state_no, 
                "%s.png"%trial_name,
            ), 
            'rb',
        )
        import matplotlib.image as mpimg
        img=mpimg.imread(vp_triangle_img)
        plot_idx = 2*trial_no+1
        ax_list[plot_idx].imshow(img)
        


    fig.set_size_inches(4*subplot_per_row,4*row_amount)

    if not os.path.isdir(figure_save_path+'/emission_log_prob_plot'):
        os.makedirs(figure_save_path+'/emission_log_prob_plot')
    title = 'state %s emission log_prob plot'%(state_no,)
    fig.savefig(os.path.join(figure_save_path, 'emission_log_prob_plot', title+".eps"), format="eps")
    fig.savefig(os.path.join(figure_save_path, 'emission_log_prob_plot', title+".png"), format="png")
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

        list_of_log_prob_mat = []
        log_prob_owner = []
        for trial_name in trials_group_by_folder_name:
            log_prob_owner.append(trial_name)

            
            emission_log_prob_mat = util.get_emission_log_prob_matrix(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )

            list_of_log_prob_mat.append(emission_log_prob_mat)

        # use np matrix to facilitate the computation of mean curve and std 
        plot_log_prob_of_all_trials(
            list_of_log_prob_mat, 
            log_prob_owner, 
            state_no, 
            figure_save_path)
