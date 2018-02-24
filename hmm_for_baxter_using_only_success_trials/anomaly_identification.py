#!/usr/bin/env python
import os
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import util
import training_config
import ipdb



def run(anomaly_data_path_for_testing,
        model_save_path,
        figure_save_path,):
    '''
        1. load all the anomalous trained models
        2. load testing anomaly data
        3. plot the log-likelihood wrt each model and plot in a same figure
    '''

    # load trained anomaly models 
    anomaly_model_group_by_label = {}
    folders = os.listdir(training_config.anomaly_model_save_path)
    for fo in folders:
        path = os.path.join(training_config.anomaly_data_path, fo)
        if not os.path.isdir(path):
            continue
        anomaly_model_path = os.path.join(training_config.anomaly_model_save_path, 
                                               fo, 
                                               training_config.config_by_user['data_type_chosen'], 
                                               training_config.config_by_user['model_type_chosen'], 
                                               training_config.model_id)
        try:
            anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
        except IOError:
            print 'anomaly model of  %s not found'%(fo,)
            continue

    # load testing anomaly data
    folders = os.listdir(anomaly_data_path_for_testing)
    for fo in folders:
        path = os.path.join(anomaly_data_path_for_testing, fo)
        if not os.path.isdir(path):
            continue
        data_path = os.path.join(anomaly_data_path_for_testing, fo)
        anomaly_testing_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, data_path, label = fo) # label = fo   
    
#---------------testing all the testing folder one by one-----------------------------------------------------------------------------------
        testing_trial_loglik_list = []
        for trial_name in anomaly_testing_group_by_folder_name:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            from matplotlib.pyplot import cm
            color = iter(cm.rainbow(np.linspace(0, 1, len(anomaly_model_group_by_label))))
            for model_label in anomaly_model_group_by_label:
                c = next(color)
                all_log_curves_of_this_model = []
                one_log_curve_of_this_model = util.fast_log_curve_calculation(anomaly_testing_group_by_folder_name[trial_name][1],
                        anomaly_model_group_by_label[model_label])
#                all_log_curves_of_this_model.append(one_log_curve_of_this_model)
                testing_trial_loglik_list.append({
                    'model_label': model_label,
                    'culmulative_loglik': one_log_curve_of_this_model[-1]})
                #--plot
                plot_line, = ax.plot(one_log_curve_of_this_model, linestyle="solid", color = c)
                plot_line.set_label('Anomaly model:' + model_label)
                ax.legend()
                title = ('Anomaly_identification for ' + fo)
                ax.set_title(title)
            sorted_loglik_list = sorted(testing_trial_loglik_list, key=lambda x:x['culmulative_loglik'])
            sorted_result = sorted_loglik_list[-1]
            cofidence = get_confidence_of_identification(sorted_result)
            ax.text(20,sorted_result['culmulative_loglik']/2, sorted_result['model_label'] + ': ' + str(cofidence),
                    ha = 'center', va = 'center',
                    bbox=dict(boxstyle="round",
                              ec=(1., 0.6, 0.6),
                              fc=(1., 0.9, 0.9),)
                    )
            if not os.path.isdir(figure_save_path + '/anomaly_identification_plot'):
                    os.makedirs(figure_save_path + '/anomaly_identification_plot')
            fig.savefig(os.path.join(figure_save_path, 'anomaly_identification_plot', fo + ":" + trial_name + ".jpg"), format="jpg")
        print 'Finish testing: '+ fo + '\n' 
    fig.show(1)

def get_confidence_of_identification(sorted_result):
    anomaly_model_path = os.path.join(training_config.anomaly_model_save_path, 
                                           sorted_result['model_label'], 
                                           training_config.config_by_user['data_type_chosen'], 
                                           training_config.config_by_user['model_type_chosen'], 
                                           training_config.model_id)
    mean_of_log_likelihood = joblib.load(os.path.join(anomaly_model_path, 'threshold_for_log_likelihood.pkl')) #mean_of_log_likelihood
    cofidence = sorted_result['culmulative_loglik']/mean_of_log_likelihood[1][-1]
    return cofidence
