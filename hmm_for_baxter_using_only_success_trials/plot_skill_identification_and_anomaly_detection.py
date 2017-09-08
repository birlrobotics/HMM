import ipdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
import os
import math

def run(
    model_save_path, 
    figure_save_path,
    state_amount,
    anomaly_detection_metric,
    trials_group_by_folder_name,
    state_order_group_by_folder_name,
    trial_class,
):

    import anomaly_detection.interface

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
        state_order = state_order_group_by_folder_name[trial_name]
        X = np.vstack([trials_group_by_folder_name[trial_name][i] for i in state_order_group_by_folder_name[trial_name]])

        skill_seq = []
        
        for t in range(0, X.shape[0]):
            now_skill, anomaly_detected, log_lik = detector.add_one_smaple_and_identify_skill_and_detect_anomaly(X[t].reshape(1,-1))
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
                
        start_t = 0
        for state_no in state_order:
            end_t = start_t+trials_group_by_folder_name[trial_name][state_no].shape[0]
            ax.axvspan(start_t, end_t, facecolor=state_color[state_no], alpha=0.5, ymax=1, ymin=0.5)
            start_t = end_t


        title = '%s detection metric %s'%(output_dir, anomaly_detection_metric)
        filename = "anoamly_detection_metric_%s"%(anomaly_detection_metric, )
        safe_filename = filename.replace("/","_divide_")
    fig.savefig(os.path.join(output_dir, safe_filename+'.eps'), format="eps")
    fig.savefig(os.path.join(output_dir, safe_filename+'.png'), format="png")






