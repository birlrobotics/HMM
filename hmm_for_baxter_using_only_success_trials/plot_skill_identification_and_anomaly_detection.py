import ipdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
import os
import math
import anomaly_detection.interface
import pandas as pd
import util

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

def color_bg(
    state_amount, 
    skill_seq, 
    ax, 
    state_idx_range_group_by_state,
    list_of_anomaly_start_idx,
    pred_start_idx_group_by_state = None,
    color_identified_skills = True,
):
    state_color = {}
    color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))
    for state_no in range(1, state_amount+1):
        state_color[state_no] = color.next()

    if color_identified_skills:
        start_t = 0
        for t in range(1, len(skill_seq)):
            if skill_seq[t-1] == skill_seq[t] and t < len(skill_seq)-1:
                continue
            skill = skill_seq[t-1]
            end_t = t

            color = util.rgba_to_rgb_using_white_bg(state_color[skill], 1)
            ax.axvspan(start_t, end_t, facecolor=color, ymax=0.5, ymin=0, lw=0)
            start_t = t

    if pred_start_idx_group_by_state is not None:
        for state_no in pred_start_idx_group_by_state:
            start_idx = pred_start_idx_group_by_state[state_no]
            ax.axvline(x=start_idx, color='black', ymax=0.5, ls='dashed')
            
            
    if color_identified_skills:
        ymin = 0.5
    else:
        ymin = 0
    for state_no in state_idx_range_group_by_state:
        state_range = state_idx_range_group_by_state[state_no]
        start_t = state_range[0]
        end_t = state_range[1]
        color = util.rgba_to_rgb_using_white_bg(state_color[state_no], 1)
        ax.axvspan(start_t, end_t, facecolor=color, ymax=1, ymin=ymin, lw=0)
        ax.axvline(x=start_t, color='black', ymax=1, ymin=0.5, ls='dashed')

    ax.axhline(y=0.5, color='black', lw=1)

    for anomaly_start_idx in list_of_anomaly_start_idx:
        ax.axvline(x=anomaly_start_idx, color='yellow')


    ax.text(-450, 0.75, 'True')
    ax.text(-450, 0.25, 'Estimate')


def get_pred_skill_start_idx(skill_seq, the_way_to_mark_skill_begins):
    if the_way_to_mark_skill_begins == 'first_occurrence':
        start_idx_group_by_state = {}
        for idx in range(len(skill_seq)):
            now_skill = skill_seq[idx]
            if now_skill not in start_idx_group_by_state:
                start_idx_group_by_state[now_skill] = idx
        return start_idx_group_by_state
    elif the_way_to_mark_skill_begins == 'first_10_successive_occurrence':
        start_idx_group_by_state = {}
        count_dict = {}
        for idx in range(len(skill_seq)):
            now_skill = skill_seq[idx]
            if now_skill not in start_idx_group_by_state:
                if now_skill not in count_dict:
                    count_dict = {}
                    count_dict[now_skill] = 1
                else:
                    count_dict[now_skill] += 1 

                if count_dict[now_skill] == 10:
                    start_idx_group_by_state[now_skill] = idx-9
                    count_dict = {}
        return start_idx_group_by_state


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
    subpolt_amount_for_each_trial = 2 
    subplot_per_row = 1
    subplot_amount = trial_amount*subpolt_amount_for_each_trial
    row_amount = int(math.ceil(float(subplot_amount)/subplot_per_row))
    fig, ax_mat = plt.subplots(nrows=row_amount, ncols=subplot_per_row)
    if row_amount == 1 and subplot_per_row == 1:
        ax_mat = np.array([ax_mat]).reshape(1, 1)
    elif row_amount == 1:
        ax_mat = ax_mat.reshape(1, -1)
    elif subplot_per_row == 1:
        ax_mat = ax_mat.reshape(-1, 1)

    ax_list = []
    for i in range(trial_amount):
        for k in range(subpolt_amount_for_each_trial):
            j = subpolt_amount_for_each_trial*i+k
            
            row_no = j/subplot_per_row
            col_no = j%subplot_per_row
            ax_list.append(ax_mat[row_no, col_no])

    trial_count = -1
    for trial_name in trials_group_by_folder_name:
        trial_count += 1

        plot_idx = trial_count*2
        ax_mark_skill_begin_by_first_occurrence = ax_list[plot_idx]
        ax_mark_skill_begin_by_first_occurrence.set_title("use first skill occurrence as skill beginning") 
        ax_mark_skill_begin_by_first_occurrence.set_yticklabels([])
        ax_mark_skill_begin_by_first_occurrence.set_xlabel("time step")

        plot_idx = trial_count*2+1
        ax_mark_skill_begin_by_first_10_successive_occurrence = ax_list[plot_idx]
        ax_mark_skill_begin_by_first_10_successive_occurrence.set_title("use first 10 successive skill occurrences as skill beginning") 
        ax_mark_skill_begin_by_first_10_successive_occurrence.set_yticklabels([])
        ax_mark_skill_begin_by_first_10_successive_occurrence.set_xlabel("time step")

        # use skill id service
        detector_using_skill_id_service = anomaly_detection.interface.get_anomaly_detector(
            model_save_path, 
            state_amount,
            anomaly_detection_metric,
        )

        print trial_name
        X = trials_group_by_folder_name[trial_name]
        skill_seq = []
        for t in range(0, X.shape[0]):
            now_skill, anomaly_detected, metric, threshold = detector_using_skill_id_service.add_one_smaple_and_identify_skill_and_detect_anomaly(X[t].reshape(1,-1))
            skill_seq.append(now_skill)

        pred_start_idx_group_by_state = get_pred_skill_start_idx(skill_seq, the_way_to_mark_skill_begins= 'first_occurrence')

        color_bg(
            state_amount, 
            skill_seq, 
            ax_mark_skill_begin_by_first_occurrence, 
            state_idx_range_by_folder_name[trial_name],
            anomaly_start_idx_group_by_folder_name[trial_name],
            pred_start_idx_group_by_state = pred_start_idx_group_by_state,
        )


        pred_start_idx_group_by_state = get_pred_skill_start_idx(skill_seq, the_way_to_mark_skill_begins= 'first_10_successive_occurrence')

        color_bg(
            state_amount, 
            skill_seq, 
            ax_mark_skill_begin_by_first_10_successive_occurrence, 
            state_idx_range_by_folder_name[trial_name],
            anomaly_start_idx_group_by_folder_name[trial_name],
            pred_start_idx_group_by_state = pred_start_idx_group_by_state,
        )


    filename = "anoamly_detection_metric_%s"%(anomaly_detection_metric, )
    safe_filename = filename.replace("/","_divide_")

    fig.set_size_inches(8*subplot_per_row,2*row_amount)

    fig. tight_layout()
    fig.savefig(os.path.join(output_dir, safe_filename+'.eps'), format="eps")
    fig.savefig(os.path.join(output_dir, safe_filename+'.png'), format="png")






