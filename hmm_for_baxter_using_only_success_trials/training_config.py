import os 
# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'wrench',
    'endpoint_pose_and_wrench'        
]
from data_fields_config_store import data_fields_store

model_type_options = [
    'hmmlearn\'s HMM', 
    'BNPY\'s HMM'
]
from model_config_store import model_store


# config provided by the user
config_by_user = {
    # config for types
    'data_type_chosen': data_type_options[0],
    'model_type_chosen': model_type_options[0],

    # config for dataset folder
    'base_path': '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170714',

    # config for preprocessing
    'preprocessing_scaling': False,
    'preprocessing_normalize': False,
    'norm_style': 'l2',

    # threshold of derivative used in hmm online anomaly detection
    'deri_threshold': 300,

    # threshold training c value in threshold=mean-c*std
    'threshold_c_value': 4
}
print 'config_by_user:', config_by_user


# auto config generation
data_type_split = config_by_user['data_type_chosen'].split("_and_")
interested_data_fields = []
for data_type in data_type_split:
    interested_data_fields += data_fields_store[data_type]
interested_data_fields.append('.tag')
print "interested_data_fields:", interested_data_fields

model_config_set_name = model_store[config_by_user['model_type_chosen']]['use']
model_config = model_store[config_by_user['model_type_chosen']]['config_set'][model_config_set_name]
print "model_config:", model_config 

def convert_camel_to_underscore(name):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
model_id = ''
for config_key in model_config:
    uncamel_key = convert_camel_to_underscore(config_key)
    for word in uncamel_key.split('_'): 
        model_id += word[0]
    model_id += '_(%s)_'%(model_config[config_key],)
print 'model_id', model_id


print '\n############'
print "press any key to continue."
raw_input()

success_path = os.path.join(config_by_user['base_path'], "success")
model_save_path = os.path.join(config_by_user['base_path'], "model", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)
figure_save_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)

exec '\n'.join("%s=%r"%i for i in config_by_user.items())



