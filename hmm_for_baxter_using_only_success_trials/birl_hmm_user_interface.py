from optparse import OptionParser
import training_config
import util

def warn(*args, **kwargs):
    if 'category' in kwargs and kwargs['category'] == DeprecationWarning:
        pass
    else:
        for arg in args:
            print arg
import warnings
warnings.warn = warn
        
def build_parser():
    parser = OptionParser()
    parser.add_option(
        "--train-model",
        action="store_true", 
        dest="train_model",
        default = False,
        help="True if you want to train HMM models.")

    parser.add_option(
        "--train-threshold",
        action="store_true", 
        dest="train_threshold",
        default = False,
        help="True if you want to train log likelihook curve threshold.")

    parser.add_option(
        "--train-derivative-threshold",
        action="store_true", 
        dest="train_derivative_threshold",
        default = False,
        help="True if you want to train derivative threshold.")

    parser.add_option(
        "--online-service",
        action="store_true", 
        dest="online_service",
        default = False,
        help="True if you want to run online anomaly detection and online state classification.")

    parser.add_option(
        "--hidden-state-log-prob-plot",
        action="store_true", 
        dest="hidden_state_log_prob_plot",
        default = False,
        help="True if you want to plot hidden state log prob.")

    parser.add_option(
        "--trial-log-likelihood-plot",
        action="store_true", 
        dest="trial_log_likelihood_plot",
        default = False,
        help="True if you want to plot trials' log likelihood.")

    parser.add_option(
        "--emission-log-prob-plot",
        action="store_true", 
        dest="emission_log_prob_plot",
        default = False,
        help="True if you want to plot emission log prob.")

    parser.add_option(
        "--trial-log-likelihood-gradient-plot",
        action="store_true", 
        dest="trial_log_likelihood_gradient_plot",
        default = False,
        help="True if you want to plot trials' log likelihood gradient.")

    parser.add_option(
        "--check-if-score-metric-converge-loglik-curves",
        action="store_true", 
        dest="check_if_score_metric_converge_loglik_curves",
        default = False,
        help="True if you want to check_if_score_metric_converge_loglik_curves.")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()

    util.inform_config(training_config)

    if options.train_model is True:
        print "gonna train HMM model."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import hmm_model_training
        hmm_model_training.run(
            model_save_path = training_config.model_save_path,
            model_type = training_config.model_type_chosen,
            model_config = training_config.model_config,
            score_metric = training_config.score_metric,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_threshold is True:
        print "gonna train threshold."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import log_likelihood_training
        log_likelihood_training.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_derivative_threshold is True:
        print "gonna train derivative threshold."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import derivative_threshold_training 
        derivative_threshold_training.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.online_service is True:
        print "gonna run online service."
        import hmm_online_service

        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)
        one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
        state_amount = len(one_trial_data_group_by_state)

        hmm_online_service.run(
            interested_data_fields = training_config.interested_data_fields,
            model_save_path = training_config.model_save_path,
            state_amount = state_amount,
            deri_threshold = training_config.deri_threshold)
            
    if options.hidden_state_log_prob_plot is True:
        print "gonna plot hidden state log prob."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import hidden_state_log_prob_plot 
        hidden_state_log_prob_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.trial_log_likelihood_plot is True:
        print "gonna plot trials' log likelihood."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import trial_log_likelihood_plot
        trial_log_likelihood_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.emission_log_prob_plot is True:
        print "gonna plot emission log prob."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import emission_log_prob_plot 
        emission_log_prob_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.trial_log_likelihood_gradient_plot is True:
        print "gonna do trial_log_likelihood_gradient_plot."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import trial_log_likelihood_gradient_plot 
        trial_log_likelihood_gradient_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.check_if_score_metric_converge_loglik_curves is True:
        print "gonna plot trials' log likelihood."
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import check_if_score_metric_converge_loglik_curves
        check_if_score_metric_converge_loglik_curves.run(
            model_save_path = training_config.model_save_path,
            model_type = training_config.model_type_chosen,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

