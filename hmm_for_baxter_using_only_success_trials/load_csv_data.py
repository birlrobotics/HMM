import os
import pandas as pd

def _load_data(path, interested_data_fields, preprocessing_normalize, preprocessing_scaling, norm=''):
    df = pd.read_csv(path, sep=',')

    df = df[interested_data_fields].loc[df['.tag'] != 0]
    state_amount = len(df['.tag'].unique())
    state_order = df['.tag'].unique().tolist()
    one_trial_data_group_by_state = {}

    # state no counts from 1
    for s in range(1, state_amount+1):
        one_trial_data_group_by_state[s] = df.loc[df['.tag'] == s].drop('.tag', axis=1).values
        if preprocessing_normalize:
            one_trial_data_group_by_state[s] = normalize(one_trial_data_group_by_state[s], norm=norm)
        if preprocessing_scaling:
            one_trial_data_group_by_state[s] = scale(one_trial_data_group_by_state[s], norm=norm)
    return one_trial_data_group_by_state, state_order

def run(data_path, interested_data_fields, preprocessing_normalize, preprocessing_scaling, norm_style=""):
    trials_group_by_folder_name = {}
    state_order_group_by_folder_name = {}

    files = os.listdir(data_path)
    for f in files:
        path = os.path.join(data_path, f)
        if not os.path.isdir(path):
            continue
        if f.startswith("bad"):
            continue

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            csv_file_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            csv_file_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have csv file."%(path,))

        one_trial_data_group_by_state, state_order = _load_data(path=csv_file_path,
                                            interested_data_fields = interested_data_fields,
                                            preprocessing_scaling=preprocessing_scaling,
                                            preprocessing_normalize=preprocessing_normalize,
                                            norm=norm_style)
        trials_group_by_folder_name[f] = one_trial_data_group_by_state
        state_order_group_by_folder_name[f] = state_order

    return trials_group_by_folder_name, state_order_group_by_folder_name
