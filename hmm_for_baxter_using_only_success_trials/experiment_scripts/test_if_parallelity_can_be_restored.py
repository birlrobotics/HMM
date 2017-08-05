import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import util
import ipdb
import os

# don't wanna see scientific notation
np.set_printoptions(suppress=True)

def tamper_input_mat(X, all_Xs):
    list_of_tampered_range = []
    length = X.shape[0]
    list_of_tampered_range.append([int(length*0.1), int(length*0.2)])
    list_of_tampered_range.append([int(length*0.5), int(length*0.6)])

    std_mat = np.array(all_Xs).std(0)

    for r in list_of_tampered_range:
        X[r[0]:r[1]+1] -= 2*std_mat[r[0]:r[1]+1]

    return X, list_of_tampered_range

def profile_model(model, output_dir, output_prefix):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy
    output_prefix = 'model_profile_'+output_prefix
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        np.savetxt(
            os.path.join(output_dir, output_prefix+'_transmat.txt'), 
            model.transmat_,
            fmt='%.6f')
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

def color_txt_lines(txt_file_path, list_of_color_range):
    txt_file = open(txt_file_path, 'r')
    lines = txt_file.readlines()
    for r in list_of_color_range:
        for i in range(r[0], r[1]+1):
            lines[i] = "\033[1;34m%s\033[0m"%(lines[i],)

    tmp_file_path = os.path.join(os.path.dirname(txt_file_path), 'tmp.txt')
    tmp_file = open(tmp_file_path, 'w')
    for l in lines:
        tmp_file.write(l)
    tmp_file.close()
    txt_file.close()

    os.rename(tmp_file_path, txt_file_path)
    

def profile_log_curve_cal(X, model, output_dir, output_prefix, list_of_color_range=[]):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy
    output_prefix = 'log_curve_cal_profile_'+output_prefix
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        np.savetxt(
            os.path.join(output_dir, output_prefix+'_framelogprob.txt'), 
            framelogprob, 
            fmt='%.6f')
        color_txt_lines(
            os.path.join(output_dir, output_prefix+'_framelogprob.txt'), 
            list_of_color_range)

        logprobij, _fwdlattice = model._do_forward_pass(framelogprob)
        np.savetxt(
            os.path.join(output_dir, output_prefix+'_fwdlattice.txt'), 
            _fwdlattice, 
            fmt='%.6f')
        color_txt_lines(
            os.path.join(output_dir, output_prefix+'_fwdlattice.txt'), 
            list_of_color_range)

    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))


def tamper_transmat(model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy
    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        hidden_state_amount = len(model.transmat_)
        model.transmat_[:] = 1.0/hidden_state_amount
        pass
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))
        

def run(model_save_path, 
    trials_group_by_folder_name,
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

    base_dir = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(base_dir, 'experiment_output', 'test_if_parallelity_can_be_restored')
    output_id = '(tamper_input)'


    if parsed_options.tamper_transmat:
        output_id += '_(tamper_transmat)'
    output_dir = os.path.join(exp_dir, output_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    for state_no in model_group_by_state:
        X = one_trial_data_group_by_state[state_no]
        all_Xs = [trials_group_by_folder_name[trial_name][state_no]\
                for trial_name in trials_group_by_folder_name]
        tampered_X, list_of_tampered_range = tamper_input_mat(X.copy(), all_Xs)

        model = model_group_by_state[state_no]
        profile_model(model, output_dir, 'state %s raw'%(state_no,))
        if parsed_options.tamper_transmat:
            tamper_transmat(model)
            profile_model(model, output_dir, 'state %s tampered'%(state_no,))
        
        log_lik_of_X = np.array(util.fast_log_curve_calculation(X, model))
        profile_log_curve_cal(X, model, output_dir, 'state %s X'%(state_no,), list_of_tampered_range)

        log_lik_of_tampered_X = np.array(util.fast_log_curve_calculation(tampered_X, model))
        profile_log_curve_cal(tampered_X, model, output_dir, 'state %s tampered_X'%(state_no,), list_of_tampered_range)





        deri_of_X = log_lik_of_X.copy()
        deri_of_X[:-1] = log_lik_of_X[1:]-log_lik_of_X[:-1]
        deri_of_X[-1] = 0

        deri_of_tampered_X = log_lik_of_tampered_X.copy()
        deri_of_tampered_X[:-1] = log_lik_of_tampered_X[1:]-log_lik_of_tampered_X[:-1]
        deri_of_tampered_X[-1] = 0

        diff = log_lik_of_X-log_lik_of_tampered_X

        deri_of_diff = diff.copy()
        deri_of_diff[:-1] = diff[1:]-diff[:-1]
        deri_of_diff[-1] = 0


        fig = plt.figure()
        ax = fig.add_subplot(411)
        title = "log lik of the two"
        ax.set_title(title)
        ax.plot(log_lik_of_X, color='black', marker='.', linestyle='None')
        ax.plot(log_lik_of_tampered_X, color='blue', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)


        ax = fig.add_subplot(412)
        title = "deri of the two"
        ax.set_title(title)
        ax.plot(deri_of_X, color='black', marker='.', linestyle='None')
        ax.plot(deri_of_tampered_X, color='blue', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)


        ax = fig.add_subplot(413)
        title = "diff of the two"
        ax.set_title(title)
        ax.plot(diff.tolist(), color='black', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)


        ax = fig.add_subplot(414)
        title = "deri of diff"
        ax.set_title(title)
        ax.plot(deri_of_diff.tolist(), color='black', marker='.', linestyle='None')
        for r in list_of_tampered_range:
            ax.axvspan(r[0], r[1], facecolor='red', alpha=0.5)


        title = "output_id %s state %s"%(output_id, state_no)
        fig.suptitle(title)



        fig.savefig(os.path.join(output_dir, title+".eps"), format="eps")
        fig.savefig(os.path.join(output_dir, title+".png"), format="png")






