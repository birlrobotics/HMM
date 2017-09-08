import ipdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 

def run(
    model_save_path, 
    figure_save_path,
    state_amount,
    anomaly_detection_metric,
    trials_group_by_folder_name,
    state_order_group_by_folder_name,
):

    import anomaly_detection.interface



    for trial_name in trials_group_by_folder_name:

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

        fig = plt.figure()        
        ax = fig.add_subplot(111)

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

        fig.show()
    raw_input()






