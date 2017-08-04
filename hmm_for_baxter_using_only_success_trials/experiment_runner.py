from optparse import OptionParser
import training_config
import util

def build_parser():
    parser = OptionParser()
    parser.add_option(
        "--test-if-parallelity-can-be-restored",
        action="store_true", 
        dest="test_if_parallelity_can_be_restored",
        default = False,
        help="read the option name please.")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()

    if options.test_if_parallelity_can_be_restored:
        print 'gonna test_if_parallelity_can_be_restored'
        trials_group_by_folder_name = util.get_trials_group_by_folder_name(training_config)

        import experiment_scripts.test_if_parallelity_can_be_restored
        experiment_scripts.test_if_parallelity_can_be_restored.run(
            model_save_path = training_config.model_save_path,
            trials_group_by_folder_name = trials_group_by_folder_name)
