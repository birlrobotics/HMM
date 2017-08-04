import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import util

def tamper_input_mat(X, all_Xs):
    list_of_tampered_range = []
    length = X.shape[0]
    list_of_tampered_range.append([int(length*0.1), int(length*0.2)])
    list_of_tampered_range.append([int(length*0.5), int(length*0.6)])

    std_mat = np.array(all_Xs).std(0)

    for r in list_of_tampered_range:
        X[r[0]:r[1]+1] -= 2*std_mat[r[0]:r[1]+1]

    return X, list_of_tampered_range

        

def run(model_save_path, 
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

    for state_no in model_group_by_state:
        X = one_trial_data_group_by_state[state_no]
        all_Xs = [trials_group_by_folder_name[trial_name][state_no]\
                for trial_name in trials_group_by_folder_name]
        tampered_X, list_of_tampered_range = tamper_input_mat(X.copy(), all_Xs)

        
        log_lik_of_X = util.fast_log_curve_calculation(
            X,
            model_group_by_state[state_no]
        )

        log_lik_of_tampered_X = util.fast_log_curve_calculation(
            tampered_X,
            model_group_by_state[state_no]
        )

        diff = np.array(log_lik_of_X)-np.array(log_lik_of_tampered_X)

        deri_of_diff = diff.copy()
        deri_of_diff[:-1] = diff[1:]-diff[:-1]
        deri_of_diff[-1] = 0


        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.set_title("log lik of the two")
        ax.plot(log_lik_of_X, color='black', marker='.', linestyle='None')
        ax.plot(log_lik_of_tampered_X, color='blue', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)

        ax = fig.add_subplot(312)
        ax.set_title("diff of the two")
        ax.plot(diff.tolist(), color='black', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)

        ax = fig.add_subplot(313)
        ax.set_title("deri of diff of the two")
        ax.plot(deri_of_diff.tolist(), color='black', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)

    plt.show()






