#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import time
import util



def assess_threshold_and_decide(
    threshold_c_value,
    mean_of_log_curve, 
    std_of_log_curve, 
    np_matrix_traj_by_time, 
    curve_owner, 
    state_no, 
    figure_save_path, 
    score_time_cost_per_point):

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # plot log curves of all trials
    for row_no in range(np_matrix_traj_by_time.shape[0]):
        trial_name = curve_owner[row_no]
        if row_no == 0:
            ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray', label='trials')
        else:
            ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray')

    # plot mean-c*std log curve
    for c in np.arange(0, 20, 2):
        ax.plot((mean_of_log_curve-c*std_of_log_curve).tolist()[0], label="mean-%s*std"%(c,), linestyle='solid')


    fig.show()

    if threshold_c_value is None:
        # decide c in an interactive way
        print "\n\nenter c (default 0.1) to visualize mean-c*std or enter ok to use this c as final threshold:"
        c = 0.1 # this is default
        while True:
            i_str = raw_input()
            if i_str == 'ok':
                title = 'state %s use threshold with c=%s (on average use %ss to compute each log likelihood point)'%(state_no, c, score_time_cost_per_point)
                ax.set_title(title)
                if not os.path.isdir(figure_save_path+'/threshold_assessment'):
                    os.makedirs(figure_save_path+'/threshold_assessment')
                fig.savefig(os.path.join(figure_save_path, 'threshold_assessment', title+".eps"), format="eps")

                plt.close(1)
                return mean_of_log_curve-c*std_of_log_curve
            try:
                c = float(i_str)
                ax.plot((mean_of_log_curve-c*std_of_log_curve).tolist()[0], label="mean-%s*std"%(c,), linestyle='dotted')
                fig.show()
            except ValueError:
                print 'bad input'
    else:
        c = threshold_c_value
        title = 'state %s use threshold with c=%s (on average use %ss to compute each log likelihood point)'%(state_no, c, score_time_cost_per_point)
        ax.set_title(title)
        if not os.path.isdir(figure_save_path+'/threshold_assessment'):
            os.makedirs(figure_save_path+'/threshold_assessment')
        fig.savefig(os.path.join(figure_save_path, 'threshold_assessment', title+".eps"), format="eps")

        plt.close(1)

        if threshold_c_value == 0:
            return mean_of_log_curve
        else:
            return mean_of_log_curve-c*std_of_log_curve
        

            
        
        
    
def run(model_save_path, 
    figure_save_path,
    threshold_c_value,
    trials_group_by_folder_name):


        
    trials_group_by_folder_name = util.make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        try:
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        except IOError:
            print 'model of state %s not found'%(state_no,)
            continue

    expected_log = []
    std_of_log = []
    threshold = []




    for state_no in model_group_by_state:
        compute_score_time_cost = 0
        total_step_times = 0


        all_log_curves_of_this_state = []
        curve_owner = []
        for trial_name in trials_group_by_folder_name:
            curve_owner.append(trial_name)
            one_log_curve_of_this_state = [] 

            start_time = time.time()
            
            one_log_curve_of_this_state = util.fast_log_curve_calculation(
                trials_group_by_folder_name[trial_name][state_no],
                model_group_by_state[state_no]
            )

            compute_score_time_cost += time.time()-start_time
            total_step_times += len(trials_group_by_folder_name[trial_name][state_no])

            all_log_curves_of_this_state.append(one_log_curve_of_this_state)

        # use np matrix to facilitate the computation of mean curve and std 
        np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)
        mean_of_log_curve = np_matrix_traj_by_time.mean(0)
        std_of_log_curve = np_matrix_traj_by_time.std(0)

        score_time_cost_per_point = float(compute_score_time_cost)/total_step_times

        decided_threshold_log_curve = assess_threshold_and_decide(
            threshold_c_value,
            mean_of_log_curve, 
            std_of_log_curve, 
            np_matrix_traj_by_time, 
            curve_owner, 
            state_no, 
            figure_save_path, 
            score_time_cost_per_point)
        expected_log.append(mean_of_log_curve.tolist()[0])
        threshold.append(decided_threshold_log_curve.tolist()[0])
        std_of_log.append(std_of_log_curve.tolist()[0])

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
        
    joblib.dump(expected_log, model_save_path+"/expected_log.pkl")
    joblib.dump(threshold, model_save_path+"/threshold.pkl")
    joblib.dump(std_of_log, model_save_path+"/std_of_log.pkl")
