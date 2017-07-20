import os 
# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'wrench',
    'endpoint_pose_and_wrench'        
]

data_fields_store = {
    "endpoint_pose": [
        '.endpoint_state.pose.position.x',
        '.endpoint_state.pose.position.y',
        '.endpoint_state.pose.position.z',
        '.endpoint_state.pose.orientation.x',
        '.endpoint_state.pose.orientation.y',
        '.endpoint_state.pose.orientation.z',
        '.endpoint_state.pose.orientation.w'
    ],
    'wrench': [
         '.wrench_stamped.wrench.force.x',
         '.wrench_stamped.wrench.force.y',
         '.wrench_stamped.wrench.force.z',
         '.wrench_stamped.wrench.torque.x',
         '.wrench_stamped.wrench.torque.y',
         '.wrench_stamped.wrench.torque.z',
    ] 
}

model_type_options = ['hmmlearn\'s HMM', 'BNPY\'s HMM']
model_config_store = {
    'hmmlearn\'s HMM': {
        'use': 'default',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'diag',
            },
        }
    },
    'BNPY\'s HMM': {
        'use': 'default',
        'config_set': {
            'default': {
                'hmm_max_train_iteration': 100,
                'hmm_hidden_state_amount': 4,
                'gaussianhmm_covariance_type_string': 'diag',
            },
        }
    }
}

# config provided by the user
config_by_user = {
    'data_type_chosen': data_type_options[0],
    'model_type_chosen': model_type_options[0],
    'base_path': '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170714',
    'preprocessing_scaling': False,
    'preprocessing_normalize': False,
    'norm_style': 'l2',
    # threshold of derivative used in hmm online anomaly detection
    'deri_threshold': 300 
}

for config_key in config_by_user:
    print config_key, ":", config_by_user[config_key]


# auto config generation
data_type_split = config_by_user['data_type_chosen'].split("_and_")
interested_data_fields = []
for data_type in data_type_split:
    interested_data_fields += data_fields_store[data_type]
interested_data_fields.append('.tag')
print "interested_data_fields:", interested_data_fields

model_config_set_name = model_config_store[config_by_user['model_type_chosen']]['use']
model_config = model_config_store[config_by_user['model_type_chosen']]['config_set'][model_config_set_name]
print "model_config:", model_config 

print '\n############'
print "press any key to continue."
raw_input()

success_path = os.path.join(config_by_user['base_path'], "success")
model_save_path = os.path.join(config_by_user['base_path'], "model", config_by_user['data_type_chosen'])
figure_save_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'])

exec '\n'.join("%s=%r"%i for i in config_by_user.items())



