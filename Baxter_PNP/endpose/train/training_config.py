import os 

interested_data_fields = [
    u'.endpoint_state.pose.position.x',
    u'.endpoint_state.pose.position.y',
    u'.endpoint_state.pose.position.z',
    u'.endpoint_state.pose.orientation.x',
    u'.endpoint_state.pose.orientation.y',
    u'.endpoint_state.pose.orientation.z',
    u'.endpoint_state.pose.orientation.w',
    u'.tag'
]

base_path = '/home/sklaw/Desktop/experiment/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_20170704_with_broken_wrench'
success_path = os.path.join(base_path, "success")
model_save_path = os.path.join(base_path, "model", "endpoint_pose")
figure_save_path = os.path.join(base_path, "figure", "endpoint_pose")

preprocessing_scaling = False
preprocessing_normalize = False
norm_style = 'l2'

hmm_max_train_iteration = 100
hmm_hidden_state_amount = 4
gaussianhmm_covariance_type_string = 'diag'
