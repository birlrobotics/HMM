import os 
import util

# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'wrench',
    'endpoint_pose_and_wrench'        
]
from data_fields_config_store import data_fields_store

model_type_options = [
    'hmmlearn\'s HMM', 
    'BNPY\'s HMM',
    'hmmlearn\'s GMMHMM',
]
from model_config_store import model_store

score_metric_options = [
    '_score_metric_last_time_stdmeanratio_',
    '_score_metric_worst_stdmeanratio_in_10_slice_',
    '_score_metric_sum_stdmeanratio_using_fast_log_cal_',
    '_score_metric_mean_of_std_using_fast_log_cal_',
    '_score_metric_hamming_distance_using_fast_log_cal_',
    '_score_metric_std_of_std_using_fast_log_cal_',
    '_score_metric_mean_of_std_divied_by_final_log_mean_',
    '_score_metric_mean_of_std_of_gradient_divied_by_final_log_mean_',
    '_score_metric_minus_diff_btw_1st_2ed_emissionprob_',
    '_score_metric_minus_diff_btw_1st_2ed(>=0)_divide_maxeprob_emissionprob_',
    '_score_metric_minus_diff_btw_1st_2ed(delete<0)_divide_maxeprob_emissionprob_',
    '_score_metric_mean_of_(std_of_(max_emissionprob_of_trials))_',
    '_score_metric_duration_of_(diff_btw_1st_2ed_emissionprob_<_10)_',
    '_score_metric_sum_of_loglik_',
]

anomaly_detection_metric_options = [
    'loglik<threshold=(mean-c*std)',
    'gradient<threshold=(min-range/2)',
    'deri_of_diff',
]

base_path_options = [
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_20170724_6states_vision (delete a bad training data)',
    '/home/sklaw/Desktop/ex/birl/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_20170724_6states_vision (delete a bad training data)',
    '/home/sklaw/Desktop/ex/birl/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170908',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170909(wrench data synced)',
    '/home/sklaw/Desktop/ex/birl/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170909_calibrated_wrench_data',
]


# config provided by the user
config_by_user = {
    # config for types
    'data_type_chosen': data_type_options[2],
    'model_type_chosen': model_type_options[0],
    'score_metric': score_metric_options[-1],
    'anomaly_detection_metric': anomaly_detection_metric_options[2],

    # config for dataset folder
    'base_path': base_path_options[-1],

    # config for preprocessing
    'preprocessing_scaling': False,
    'preprocessing_normalize': False,
    'norm_style': 'l2',

    # threshold of derivative used in hmm online anomaly detection
    'deri_threshold': 200,

    # threshold training c value in threshold=mean-c*std
    'threshold_c_value': 20
}

# auto config generation
data_type_split = config_by_user['data_type_chosen'].split("_and_")
interested_data_fields = []
for data_type in data_type_split:
    interested_data_fields += data_fields_store[data_type]
interested_data_fields.append('.tag')

model_config_set_name = model_store[config_by_user['model_type_chosen']]['use']
model_config = model_store[config_by_user['model_type_chosen']]['config_set'][model_config_set_name]

model_id     = util.get_model_config_id(model_config)
model_id     = config_by_user['score_metric']+model_id


success_path = os.path.join(config_by_user['base_path'], "success")
anomaly_data_path = os.path.join(config_by_user['base_path'], "has_anomaly")
test_success_data_path = os.path.join(config_by_user['base_path'], "success_for_test")
model_save_path = os.path.join(config_by_user['base_path'], "model", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)
anomaly_model_save_path = os.path.join(config_by_user['base_path'], "anomaly_model", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)
figure_save_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)

exec '\n'.join("%s=%r"%i for i in config_by_user.items())



