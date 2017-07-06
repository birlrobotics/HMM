from optparse import OptionParser
import training_config


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
    usage = "usage: %prog --train-model --train-threshold --run-online-detection"
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
            n_state = training_config.hmm_hidden_state_amount,
            covariance_type_string = training_config.gaussianhmm_covariance_type_string,
            n_iteration = training_config.hmm_max_train_iteration,
            trials_group_by_folder_name = trials_group_by_folder_name)

    if options.train_threshold is True:
        print "gonna train threshold."
        trials_group_by_folder_name = get_trials_group_by_folder_name()

        import log_likelihood_training
        log_likelihood_training.run(
            model_save_path = training_config.model_save_path,
            figure_save_path = training_config.figure_save_path,
            trials_group_by_folder_name = trials_group_by_folder_name)
            
        










