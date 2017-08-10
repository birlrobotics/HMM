from sklearn.externals import joblib
import util
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
import os

def color_bg_by_state(state_order, state_color, state_start_idx, ax):
    for idx in range(len(state_start_idx)-1):
        start_at = state_start_idx[idx]
        end_at = state_start_idx[idx+1]
        ax.axvspan(start_at, end_at, facecolor=state_color[state_order[idx]], alpha=0.25)

def run(model_save_path, 
    figure_save_path,
    trials_group_by_folder_name,
    state_order_group_by_folder_name,
    parsed_options):

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

    state_color = {}
    color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))
    for state_no in model_group_by_state:
        state_color[state_no] = color.next()


    output_dir = os.path.join(figure_save_path, 'test_if_gradient_can_detect_state_switch')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for trial_name in trials_group_by_folder_name:
        X = None

        state_start_idx = [0]

        state_order = state_order_group_by_folder_name[trial_name]
        for state_no in state_order:
            if X is None:
                X = trials_group_by_folder_name[trial_name][state_no]
            else:
                X = np.concatenate((X, trials_group_by_folder_name[trial_name][state_no]),axis = 0)
            state_start_idx.append(len(X))

        fig = plt.figure()
        ax_loglik = fig.add_subplot(211)
        ax_loglik_gradient = fig.add_subplot(212)
        bbox_extra_artists = []
        for state_no in model_group_by_state:
            log_lik_curve = np.array(util.fast_log_curve_calculation(
                X,
                model_group_by_state[state_no]
            ))

            log_lik_gradient_curve = log_lik_curve[1:]-log_lik_curve[:-1]
            ax_loglik.plot(log_lik_curve, label='state %s'%(state_no,), color=state_color[state_no])
            ax_loglik_gradient.plot(np.log(log_lik_gradient_curve), label='state %s'%(state_no,), color=state_color[state_no])


        color_bg_by_state(state_order, state_color, state_start_idx, ax_loglik)
        color_bg_by_state(state_order, state_color, state_start_idx, ax_loglik_gradient)

        title = "trial %s loglik of all states"%(trial_name,)
        ax_loglik.set_title(title)
        title = "trial %s loglik gradient of all states"%(trial_name,)
        ax_loglik_gradient.set_title(title)

        lgd = ax_loglik.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)
        lgd = ax_loglik_gradient.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)

        title = "trial %s"%(trial_name,)
        fig.savefig(os.path.join(output_dir, title+".eps"), format="eps", bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, title+".png"), format="png", bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
