#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from hmmlearn.hmm import *
from sklearn.externals import joblib
from math import (
    log,
    exp
)
from sklearn.preprocessing import (
    scale,
    normalize
)

import time 

import training_config

# globala variables
state_amount = None


def matplot_list(list_data,
                 figure_index,
                 title,
                 label_string,
                 save_path,
                 save=False,
                 linewidth='3.0',
                 fontsize= 50,
                 xaxis_interval=0.01,
                 xlabel= 'time',
                 ylabel = 'value'):
    # if you want to save, title is necessary as a save name.
    
    global n_state
    global covariance_type_string
    plt.figure(figure_index, figsize=(40,30), dpi=80)
    ax = plt.subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    plt.grid(True)
    i = 0
    plt.xlabel(xlabel,fontsize=fontsize)
  

    plt.ylabel(ylabel,fontsize=fontsize)

    plt.xticks( fontsize = 50)
    plt.yticks( fontsize = 50)
    
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])*xaxis_interval).tolist()
        if label_string[i-1] == 'mean+std'or label_string[i-1]=='mean-std':
            plt.plot(O, data, label=label_string[i-1],linewidth=3, linestyle = '--',color ="grey")
        else:
            plt.plot(O, data, label=label_string[i-1],linewidth=linewidth)
    plt.legend(loc='best', frameon=True, fontsize=fontsize)

    plt.title(title, fontsize=fontsize)

    #plt.annotate('State=4 Sub_State='+str(n_state)+' GaussianHMM_cov='+covariance_type_string,
    #         xy=(0, -5000), xycoords='data',
    #         xytext=(+10, +30), textcoords='offset points', fontsize=fontsize,
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.25"))

    if save:
        plt.savefig(save_path+'/'+title+".eps", format="eps")


def scaling(X):
    _index, _column = X.shape
    Data_scaled = []
    scale_length = 10
    for i in range(scale_length, _index-scale_length-2):
        scaled = scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array = np.asarray(Data_scaled)
    return scaled_array
    


def load_data(path, preprocessing_normalize, preprocessing_scaling=False, norm='l2'):
    global state_amount
    df = pd.read_csv(path, sep=',')

    df = df[training_config.interested_data_fields].loc[df['.tag'] != 0]
    state_amount = df.tail(1)['.tag']
    one_trial_data_group_by_state = {}

    # state no counts from 1
    for s in range(1, state_amount+1):
        one_trial_data_group_by_state[s] = df.loc[df['.tag'] == s].drop('.tag', axis=1).values
        if preprocessing_normalize:
            one_trial_data_group_by_state[s] = normalize(one_trial_data_group_by_state[s], norm=norm)
        if preprocessing_scaling:
            one_trial_data_group_by_state[s] = scale(one_trial_data_group_by_state[s], norm=norm)
    return one_trial_data_group_by_state

def array_list_mean(list_data):
    """
    eg: argument list_data[X1,X2,...]
        mean(X1,X2,..)    

    return: marray [X_mean] , [X_std]
    """
    tempt_list = []
    df_full_mean = pd.DataFrame()
    df_full_std = pd.DataFrame()
    df = pd.DataFrame()
    if len(list_data[0].shape) ==1:
        for data in list_data:
            df_tempt = pd.DataFrame(data=data)
            df = pd.concat([df,df_tempt], axis=1)
        mean_series = df.mean(axis=1)
        std_series = df.std(axis=1)
        df_full_mean = pd.concat([df_full_mean,mean_series], axis=1)
        df_full_std = pd.concat([df_full_std,std_series], axis=1)
    else:
        index, column = list_data[0].shape()
        for i in range(column):
            for data in list_data:
                df_tempt = pd.DataFrame(data=data[:,i])
                df = pd.concat([df,df_tempt], axis=1)
            mean_series = df.mean(axis=1)
            std_series = df.std(axis=1)
            df_full_mean = pd.concat([df_full_mean,mean_series], axis=1)
            df_full_std = pd.concat([df_full_std,std_series], axis=1)
    return df_full_mean.values, df_full_std.values

    
def main():


    global n_state
    global covariance_type_string

    #substate for every hmm model
    n_state = 4

    n_iteraton = 100

    covariance_type_string = 'diag'

    preprocessing_scaling = False

    preprocessing_normalize = False

    data_feature = 7

    norm_style = 'l2'

    base_path = '/home/sklaw/Desktop/experiment/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_20170704_with_broken_wrench'

    success_path = os.path.join(base_path, "success")

    model_save_path = os.path.join(base_path, "model", "endpoint_pose")

    figure_save_path = os.path.join(base_path, "figure", "endpoint_pose")

    data_feature_name = ['position_x',
                         'position_y',
                         'position_z',
                         'orientation_x',
                         'orientation_y',
                         'orientation_z',
                         'orientation_w']


    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)

    success_trial_amount = 0
    list_of_trials = []

    files = os.listdir(success_path)
    for f in files:
        path = os.path.join(success_path, f)
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

        success_trial_amount += 1
        one_trial_data_group_by_state = load_data(path=csv_file_path,
                                            preprocessing_scaling=preprocessing_scaling,
                                            preprocessing_normalize=preprocessing_normalize,
                                            norm=norm_style)
        list_of_trials.append(one_trial_data_group_by_state)

    start_prob = np.zeros(n_state)
    start_prob[0] = 1

    training_data_group_by_state = {}
    training_length_array_group_by_state = {}

    for state_no in range(1, state_amount+1):
        length_array = []
        for trial_no in range(len(list_of_trials)):
            length_array.append(list_of_trials[trial_no][state_no].shape[0])
            if trial_no == 0:
                data_tempt = list_of_trials[trial_no][state_no]
            else:
                data_tempt = np.concatenate((data_tempt,list_of_trials[trial_no][state_no]),axis = 0)
        training_data_group_by_state[state_no] = data_tempt
        training_length_array_group_by_state[state_no] = length_array

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        model = GaussianHMM(
            n_components=n_state, 
            covariance_type=covariance_type_string,
            params="mct", 
            init_params="cmt", 
            n_iter=n_iteraton)
        model.startprob_ = start_prob
        model = model.fit(
            training_data_group_by_state[state_no],
            lengths=training_length_array_group_by_state[state_no])
    
        model_group_by_state[state_no] = model


    # save the models
    if not os.path.isdir(model_save_path+"/multisequence_model"):
        os.makedirs(model_save_path+"/multisequence_model")
    for state_no in range(1, state_amount+1):
        joblib.dump(
            model_group_by_state[state_no], 
            model_save_path+"/multisequence_model/model_s%s.pkl"%(state_no,))
    
    joblib.dump(n_state, model_save_path+"/multisequence_model/n_state.pkl")
    joblib.dump(covariance_type_string, model_save_path+"/multisequence_model/covariance_type.pkl")

if __name__ == '__main__':
    sys.exit(main())
