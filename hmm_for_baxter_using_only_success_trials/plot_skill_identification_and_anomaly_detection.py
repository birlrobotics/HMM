import ipdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
import os
import math
import anomaly_detection.interface
import pandas as pd

def load_data(
    data_path, 
    interested_data_fields,
):
    trials_group_by_folder_name = {}
    state_idx_range_by_folder_name = {}
    anomaly_start_idx_group_by_folder_name = {}

    files = os.listdir(data_path)
    for f in files:
        path = os.path.join(data_path, f)
        if not os.path.isdir(path):
            continue

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            tm_csv_file_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            tm_csv_file_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have csv file."%(path,))

        tag_multimodal_df = pd.read_csv(tm_csv_file_path, sep=',')
        tag_multimodal_df = tag_multimodal_df.loc[tag_multimodal_df['.tag'] != 0]
        tag_multimodal_df.index = np.arange(0, len(tag_multimodal_df))

        state_idx_range_by_folder_name[f] = {}
        list_of_state = tag_multimodal_df['.tag'].unique()
        for state_no in list_of_state:
            state_df = tag_multimodal_df[tag_multimodal_df['.tag'] == state_no]
            start_idx = state_df.head(1).index[0]
            end_idx = state_df.tail(1).index[0]
            state_idx_range_by_folder_name[f][state_no] = [start_idx, end_idx]

        trials_group_by_folder_name[f] = tag_multimodal_df[interested_data_fields].drop('.tag', axis=1).values

        if os.path.isfile(os.path.join(path, f+'-anomaly_timestamp.csv')):
            at_csv_file_path = os.path.join(path, f+'-anomaly_timestamp.csv')
        else:
            at_csv_file_path = None

        anomaly_start_idx_group_by_folder_name[f] = []
        if at_csv_file_path is None:
            pass
        else:
            at_df = pd.read_csv(at_csv_file_path, sep=',')

            from dateutil import parser
            tag_multimodal_df['time'] = tag_multimodal_df['time'].apply(lambda x: parser.parse(x))
            at_df['time'] = at_df['time'].apply(lambda x: parser.parse(x))

            for at_idx in range(at_df.shape[0]):
                anomaly_start_time = at_df['time'][at_idx]
                tm_idx = tag_multimodal_df[tag_multimodal_df['time']>anomaly_start_time].head(1).index[0]
                anomaly_start_idx_group_by_folder_name[f].append(tm_idx)

    return trials_group_by_folder_name, state_idx_range_by_folder_name, anomaly_start_idx_group_by_folder_name

def run(
    model_save_path, 
    figure_save_path,
    anomaly_detection_metric,
    trial_class,
    data_path,
    interested_data_fields,
):

    trials_group_by_folder_name, state_idx_range_by_folder_name, anomaly_start_idx_group_by_folder_name = load_data(data_path, interested_data_fields,)
    
    state_amount = len(state_idx_range_by_folder_name.itervalues().next())

    output_dir = os.path.join(
        figure_save_path,
        "anomaly_detection_assessment",
        trial_class,
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    trial_amount = len(trials_group_by_folder_name)

    subplot_per_row = 2 
    subplot_amount = trial_amount
    row_amount = int(math.ceil(float(subplot_amount)/subplot_per_row))
    fig, ax_mat = plt.subplots(nrows=row_amount, ncols=subplot_per_row)
    fig.set_size_inches(8*subplot_per_row,8*row_amount)
    if row_amount == 1:
        ax_mat = ax_mat.reshape(1, -1)

    count = 0
    for trial_name in trials_group_by_folder_name:
        row_no = count/subplot_per_row
        col_no = count%subplot_per_row
        count += 1

        ax = ax_mat[row_no, col_no]
        ax.set_title(trial_name)

        detector = anomaly_detection.interface.get_anomaly_detector(
            model_save_path, 
            state_amount,
            anomaly_detection_metric,
        )

        print trial_name
        X = trials_group_by_folder_name[trial_name]
        skill_seq = []
        for t in range(0, X.shape[0]):
            now_skill, anomaly_detected, metric, threshold = detector.add_one_smaple_and_identify_skill_and_detect_anomaly(X[t].reshape(1,-1))
            skill_seq.append(now_skill)

        detector.plot_metric_data(ax)

        state_color = {}
        color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))
        for state_no in range(1, state_amount+1):
            state_color[state_no] = color.next()

        start_t = 0
        for t in range(1, X.shape[0]):
            if skill_seq[t-1] == skill_seq[t] and t < X.shape[0]-1:
                continue
            skill = skill_seq[t-1]
            end_t = t
            ax.axvspan(start_t, end_t, facecolor=state_color[skill], alpha=0.25, ymax=0.5, ymin=0)
            start_t = t
                
        for state_no in state_idx_range_by_folder_name[trial_name]:
            state_range = state_idx_range_by_folder_name[trial_name][state_no]
            start_t = state_range[0]
            end_t = state_range[1]
            ax.axvspan(start_t, end_t, facecolor=state_color[state_no], alpha=0.5, ymax=1, ymin=0.5)

        for anomaly_start_idx in anomaly_start_idx_group_by_folder_name[trial_name]:
            ax.axvline(x=anomaly_start_idx, color='yellow')


        title = '%s detection metric %s'%(output_dir, anomaly_detection_metric)
        filename = "anoamly_detection_metric_%s"%(anomaly_detection_metric, )
        safe_filename = filename.replace("/","_divide_")
    fig.savefig(os.path.join(output_dir, safe_filename+'.eps'), format="eps")
    fig.savefig(os.path.join(output_dir, safe_filename+'.png'), format="png")






