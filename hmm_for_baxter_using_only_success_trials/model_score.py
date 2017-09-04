import util
import numpy as np
import ipdb

def score(score_metric, model, X, lengths):
    if score_metric == '_score_metric_worst_stdmeanratio_in_10_slice_':
        slice_10_time_step_log_lik = [[model.score(X[i:i+k*(j-i)/10]) for k in range(1, 11, 1)] for i, j in util.iter_from_X_lengths(X, lengths)]
        matrix = np.matrix(slice_10_time_step_log_lik)
        slice_10_means = abs(matrix.mean(0))
        slice_10_std = matrix.std(0)
        slice_10_stme_ratio = slice_10_std/slice_10_means
        score = slice_10_stme_ratio.max()
    elif score_metric == '_score_metric_last_time_stdmeanratio_':
        final_time_step_log_lik = [
            model.score(X[i:j]) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        matrix = np.matrix(final_time_step_log_lik)
        mean = abs(matrix.mean())
        std = matrix.std()
        score = std/mean
    elif score_metric == '_score_metric_sum_stdmeanratio_using_fast_log_cal_':
        final_time_step_log_lik = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(final_time_step_log_lik) 
        mean_of_log_curve = curve_mat.mean(0)
        std_of_log_curve = curve_mat.std(0)
        score = abs(std_of_log_curve/mean_of_log_curve).mean()
    elif score_metric == '_score_metric_mean_of_std_using_fast_log_cal_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        std_of_log_curve = curve_mat.std(0)
        score = std_of_log_curve.mean()
    elif score_metric == '_score_metric_hamming_distance_using_fast_log_cal_':
        import scipy.spatial.distance as sp_dist
        log_lik = [util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        log_mat         = np.matrix(log_lik)
        std_of_log_mat  = log_mat.std(0)
        mean_of_log_mat = log_mat.mean(0)
        lower_bound     = mean_of_log_mat - 20 * std_of_log_mat
        ipdb.set_trace()
        hamming_score   = sp_dist.hamming(mean_of_log_mat, lower_bound)
        score  = hamming_score
    elif score_metric == '_score_metric_std_of_std_using_fast_log_cal_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        std_of_log_curve = curve_mat.std(0)
        score = std_of_log_curve.std()
    elif score_metric == '_score_metric_mean_of_std_divied_by_final_log_mean_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        std_of_log_curve = curve_mat.std(0)
        mean_of_std = std_of_log_curve.mean()
        final_log_mean = curve_mat.mean(0)[0, -1]
        score = abs(mean_of_std/final_log_mean)
    elif score_metric == '_score_metric_mean_of_std_of_gradient_divied_by_final_log_mean_':
        log_curves_of_all_trials = [
            util.fast_log_curve_calculation(X[i:j], model) for i, j in util.iter_from_X_lengths(X, lengths)
        ]
        
        curve_mat = np.matrix(log_curves_of_all_trials) 
        gradient_mat = curve_mat[:, 1:]-curve_mat[:, :-1]
        std_of_log_curve = gradient_mat.std(0)
        mean_of_std = std_of_log_curve.mean()
        final_log_mean = gradient_mat.mean(0)[0, -1]
        score = abs(mean_of_std/final_log_mean)
    else:
        raise Exception('unknown score metric \'%s\''%(score_metric,))

    return score
