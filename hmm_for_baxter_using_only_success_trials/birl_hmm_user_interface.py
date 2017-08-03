from optparse import OptionParser
import training_config

def warn(*args, **kwargs):
    if 'category' in kwargs and kwargs['category'] == DeprecationWarning:
        pass
    else:
        for arg in args:
            print arg
import warnings
warnings.warn = warn

def get_trials_group_by_folder_name():
    import copy
    if (get_trials_group_by_folder_name.done):
        return copy.deepcopy(get_trials_group_by_folder_name.trials_group_by_folder_name)


    import load_csv_data
    trials_group_by_folder_name = load_csv_data.run(
        success_path = training_config.success_path,
        interested_data_fields = training_config.interested_data_fields,
        preprocessing_normalize = training_config.preprocessing_normalize,
        preprocessing_scaling = training_config.preprocessing_scaling
    )

    get_trials_group_by_folder_name.done = True
    get_trials_group_by_folder_name.trials_group_by_folder_name = trials_group_by_folder_name
    return copy.deepcopy(get_trials_group_by_folder_name.trials_group_by_folder_name)
        
def build_parser():
    usage = "usage: %prog --train-model --train-threshold --online-service"
    parser = OptionParser(usage=usage)
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
        "--hidden-state-pmf-plot",
        action="store_true", 
        dest="hidden_state_pmf_plot",
        default = False,
        help="True if you want to plot hidden state pmf.")

    return parser

if __name__ == "__main__":
    get_trials_group_by_folder_name.done = False

    parser = build_parser()
    (options, args) = parser.parse_args()

    if options.train_model is True:
        print "gonna train HMM model."
        trials_group_by_folder_name = get_trials_group_by_folder_name()

        import hmm_model_training
        hmm_model_training.run(
            model_save_path = training_config.model_save_path,
            model_type = training_config.model_type_chosen,
            model_config = training_config.model_config,
            score_metric = training_config.score_metric,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_threshold is True:
        print "gonna train threshold."
        trials_group_by_folder_name = get_trials_group_by_folder_name()

        import log_likelihood_training
        log_likelihood_training.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_derivative_threshold is True:
        print "gonna train derivative threshold."
        trials_group_by_folder_name = get_trials_group_by_folder_name()

        import derivative_threshold_training 
        derivative_threshold_training.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.online_service is True:
        print "gonna run online service."
        import hmm_online_service

        trials_group_by_folder_name = get_trials_group_by_folder_name()
        one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
        state_amount = len(one_trial_data_group_by_state)

        hmm_online_service.run(
            interested_data_fields = training_config.interested_data_fields,
            model_save_path = training_config.model_save_path,
            state_amount = state_amount,
            deri_threshold = training_config.deri_threshold)
            
    if options.hidden_state_pmf_plot is True:
        print "gonna plot hidden state pmf."
        trials_group_by_folder_name = get_trials_group_by_folder_name()

        import hidden_state_pmf_plot 
        hidden_state_pmf_plot.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            threshold_c_value = training_config.threshold_c_value,
            trials_group_by_folder_name = trials_group_by_folder_name)
