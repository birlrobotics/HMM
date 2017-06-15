#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from hmmlearn.hmm import *
from sklearn.externals import joblib
import ipdb
from math import (
    log,
    exp
)
from sklearn.preprocessing import (
    scale,
    normalize
)

import time 

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
    


def load_data(path, preprocessing_scaling=False, preprocessing_normalize=False, norm='l2'):
    """
       df.columns = u'time', u'.endpoint_state.header.seq',
       u'.endpoint_state.header.stamp.secs',
       u'.endpoint_state.header.stamp.nsecs',
       u'.endpoint_state.header.frame_id', u'.endpoint_state.pose.position.x',
       u'.endpoint_state.pose.position.y', u'.endpoint_state.pose.position.z',
       u'.endpoint_state.pose.orientation.x',
       u'.endpoint_state.pose.orientation.y',
       u'.endpoint_state.pose.orientation.z',
       u'.endpoint_state.pose.orientation.w',
       u'.endpoint_state.twist.linear.x', u'.endpoint_state.twist.linear.y',
       u'.endpoint_state.twist.linear.z', u'.endpoint_state.twist.angular.x',
       u'.endpoint_state.twist.angular.y', u'.endpoint_state.twist.angular.z',
       u'.endpoint_state.wrench.force.x', u'.endpoint_state.wrench.force.y',
       u'.endpoint_state.wrench.force.z', u'.endpoint_state.wrench.torque.x',
       u'.endpoint_state.wrench.torque.y', u'.endpoint_state.wrench.torque.z',
       u'.joint_state.header.seq', u'.joint_state.header.stamp.secs',
       u'.joint_state.header.stamp.nsecs', u'.joint_state.header.frame_id',
       u'.joint_state.name', u'.joint_state.position',
       u'.joint_state.velocity', u'.joint_state.effort',
       u'.wrench_stamped.header.seq', u'.wrench_stamped.header.stamp.secs',
       u'.wrench_stamped.header.stamp.nsecs',
       u'.wrench_stamped.header.frame_id', u'.wrench_stamped.wrench.force.x',
       u'.wrench_stamped.wrench.force.y', u'.wrench_stamped.wrench.force.z',
       u'.wrench_stamped.wrench.torque.x', u'.wrench_stamped.wrench.torque.y',
       u'.wrench_stamped.wrench.torque.z', u'.tag']
    """
    df = pd.read_csv(path+"/tag_multimodal.csv",sep=',')

    df = df[[u'.wrench_stamped.wrench.force.x',
             u'.wrench_stamped.wrench.force.y',
             u'.wrench_stamped.wrench.force.z',
             u'.wrench_stamped.wrench.torque.x',
             u'.wrench_stamped.wrench.torque.y',
             u'.wrench_stamped.wrench.torque.z',
             u'.tag']]
 
 
    X_1 = df.values[df.values[:,-1] ==1]
    index_1,column_1 = X_1.shape
    X_2 = df.values[df.values[:,-1] ==2]
    index_2,column_2 = X_2.shape
    X_3 = df.values[df.values[:,-1] ==3]
    index_3,column_3 = X_3.shape
    X_4 = df.values[df.values[:,-1] ==4]
    index_4,column_4 = X_4.shape
    #X_5 = df.values[df.values[:,-1] ==5]
    #index_5,column_5 = X_5.shape
    
    index = [index_1,index_2,index_3,index_4]

    X_1_ = X_1[:,:-1]
    X_2_ = X_2[:,:-1]
    X_3_ = X_3[:,:-1]
    X_4_ = X_4[:,:-1]
    #X_5_ = X_5[:,:-1]
    
    Data = [X_1_,X_2_,X_3_,X_4_]

    if preprocessing_normalize:
        normalize_X_1 = normalize(Data[0], norm=norm)
        normalize_X_2 = normalize(Data[1], norm=norm)
        normalize_X_3 = normalize(Data[2], norm=norm)
        normalize_X_4 = normalize(Data[3], norm=norm)
        normalize_X_5 = normalize(Data[4], norm=norm)
        
        index_1, column_1 = normalize_X_1.shape
        index_2, column_2 = normalize_X_2.shape
        index_3, column_3 = normalize_X_3.shape
        index_4, column_4 = normalize_X_4.shape
        index_5, column_5 = normalize_X_5.shape

        Data = []
        Data = [normalize_X_1, normalize_X_2, normalize_X_3, normalize_X_4, normalize_X_5]
        index = [index_1, index_2, index_3, index_4, index_5]
        
    if preprocessing_scaling:
        scaled_X_1 = scale(Data[0])
        scaled_X_2 = scale(Data[1])
        scaled_X_3 = scale(Data[2])
        scaled_X_4 = scale(Data[3])
        scaled_X_5 = scale(Data[4])

        index_1, column_1 = scaled_X_1.shape
        index_2, column_2 = scaled_X_2.shape
        index_3, column_3 = scaled_X_3.shape
        index_4, column_4 = scaled_X_4.shape
        index_5, column_5 = scaled_X_5.shape

        Data = []
        Data = [scaled_X_1, scaled_X_2, scaled_X_3, scaled_X_4, scaled_X_5]
        
        index = [index_1,index_2,index_3,index_4, index_5]
        
    return Data, index

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

    #ipdb.set_trace()

    global n_state
    global covariance_type_string

    #substate for every hmm model
    n_state = 6

    n_iteraton = 100

    covariance_type_string = 'diag'

    preprocessing_scaling = False

    preprocessing_normalize = False

    data_feature = 6

    norm_style = 'l2'

    success_path = "/home/ben/ML_data/REAL_BAXTER_PICK_N_PLACE_6_1/success"

    model_save_path = "/home/ben/ML_data/REAL_BAXTER_PICK_N_PLACE_6_1/model/wrench"

    figure_save_path = "/home/ben/ML_data/REAL_BAXTER_PICK_N_PLACE_6_1/figure/wrench"

    success_train_num = 10

    data_feature_name = ['Force X',
                         'Force Y',
                         'Force Z',
                         'Torque X',
                         'Torque Y',
                         'Torque Z']


    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)


    #failure_test_path = "/home/ben/ML_data/SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob/XX+r0.1968"

    
    # load the success Data Index And Label String
    path_index_name = []


    for i in range(success_train_num):
        if i+1 <= 9:
            post_str = '0'+str(i+1)
        else:
            post_str = str(i+1)
            
        path_index_name.append(post_str)
        
    Success_Data = []
    Success_Index = []
    Success_Label_String = []
    for i in range(success_train_num):
        data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i],
                                            preprocessing_scaling=preprocessing_scaling,
                                            preprocessing_normalize=preprocessing_normalize,
                                            norm=norm_style)
        Success_Data.append(data_tempt)
        Success_Index.append(index_tempt)
        Success_Label_String.append("Success Trail 0"+ str(i+1))



        
    # get the FX data list in State 2 and get the data mean and std
    ##  Success Data [Trails] [Subtask] [index,columns]

    # if not os.path.isdir(figure_save_path+"/original_data"):
    #     os.makedirs(figure_save_path+"/original_data")
    # for n in range(4):
    #     if not os.path.isdir(figure_save_path+"/original_data/substask_"+str(n+1)):
    #         os.makedirs(figure_save_path+"/original_data/substask_"+str(n+1))
    #     for f in range(data_feature):
    #         success_data_fx_list = []
    #         success_data_string = []
    #         #success_data_fx_string_list = Success_Label_String
    #         for i in range(success_train_num):
    #             success_data_fx_list.append(Success_Data[i][n][:,f])
                
    #         success_S2_mean_data, success_S2_std_data = array_list_mean(success_data_fx_list)
                
    #         success_data_fx_list = []
    #         #for i in range(success_train_num):
    #         #success_data_fx_list.append(Success_Data[i][1][:,f].T)
                
    #         success_data_fx_list.append(success_S2_mean_data.T[0,:])
    #         success_data_fx_list.append(success_S2_mean_data.T[0,:]+success_S2_std_data.T[0,:])
    #         success_data_fx_list.append(success_S2_mean_data.T[0,:]-success_S2_std_data.T[0,:])
    #         success_data_string.append("mean")
    #         success_data_string.append("mean+std")
    #         success_data_string.append("mean-std")
            
    #         matplot_list(success_data_fx_list,
    #                      figure_index=f+n*10,
    #                      title="Subtask "+str(n)+" Data "+ data_feature_name[f]+" Mean and std",
    #                      save=True,
    #                      save_path=figure_save_path+"/original_data/substask_"+str(n+1),
    #                      label_string=success_data_string)

    #plt.show()
    

    ## averaging the data 
    # success_S1_mean_data, success_S1_std_data = array_list_mean(success_S1_full_data_list)
    
    # success_S2_mean_data, success_S2_std_data = array_list_mean(success_S2_full_data_list)
 
    # success_S3_mean_data, success_S3_std_data = array_list_mean(success_S3_full_data_list)
    
    # success_S4_mean_data, success_S4_std_data = array_list_mean(success_S3_full_data_list)
    
    # success_S5_mean_data, success_S5_std_data = array_list_mean(success_S3_full_data_list)

    # train_Data = [success_S1_mean_data,
    #               success_S2_mean_data,
    #               success_S3_mean_data,
    #               success_S4_mean_data,
    #               success_S5_mean_data]
    # index1, column1 = success_S1_mean_data.shape
    # index2, column2 = success_S2_mean_data.shape
    # index3, column3 = success_S3_mean_data.shape
    # index4, column4 = success_S4_mean_data.shape
    # index5, column5 = success_S5_mean_data.shape
    # train_index = [index1,index2,index3,index4,index5]

    
    
    #initial start probalility    
    start_prob = np.zeros(n_state)
    #init_trans_mat = np.zeros((n_state,n_state))
    # for i in range(n_state):
    #     init_trans_mat[i][i] = 0.8
    #     if not i== n_state - 1:
    #         init_trans_mat[i][i+1] = 0.2
    #     else:
    #         init_trans_mat[i][i] = 1.0
            
    start_prob[0] = 1
    # Subtasks i hmm model list
    model_1_list = []
    model_2_list = []
    model_3_list = []
    model_4_list = []

    train_Data = []
    train_length = []
    train_length_tempt = []

    for i in range(success_train_num):
        if i==0:
            data_tempt = Success_Data[i][0]
            train_length_tempt = []
        else:
            data_tempt = np.concatenate((data_tempt,Success_Data[i][0]),axis = 0)
        train_length_tempt.append(Success_Index[i][0])
    train_Data.append(data_tempt)
    train_length.append(train_length_tempt)

    
    for i in range(success_train_num):
        if i==0:
            data_tempt = Success_Data[i][1]
            train_length_tempt = []
        else:
            data_tempt = np.concatenate((data_tempt,Success_Data[i][1]),axis = 0)
        train_length_tempt.append(Success_Index[i][1])
    train_Data.append(data_tempt)
    train_length.append(train_length_tempt)

    
    for i in range(success_train_num):
        if i==0:
            data_tempt = Success_Data[i][2]
            train_length_tempt = []
        else:
            data_tempt = np.concatenate((data_tempt,Success_Data[i][2]),axis = 0)
        train_length_tempt.append(Success_Index[i][2])
    train_Data.append(data_tempt)
    train_length.append(train_length_tempt)

    
    for i in range(success_train_num):
        if i==0:
            data_tempt = Success_Data[i][3]
            train_length_tempt = []
        else:
            data_tempt = np.concatenate((data_tempt,Success_Data[i][3]),axis = 0)
        train_length_tempt.append(Success_Index[i][3])
    train_Data.append(data_tempt)
    train_length.append(train_length_tempt)
        
 
    model_1 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=n_iteraton)
    model_1.startprob_ = start_prob
    #model_1.transmat_ = init_trans_mat
    model_1 = model_1.fit(train_Data[0],lengths=train_length[0])
    try:
        log_tempt = model_1.score(Success_Data[0][0])
    except:
        print"train %d model 1"%(i+1)
        return 0
        
    model_1_list.append(model_1)


    model_2 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=n_iteraton)
    model_2.startprob_ = start_prob
    #model_2.transmat_ = init_trans_mat
    model_2 = model_2.fit(train_Data[1],lengths=train_length[1])
    try:
        log_tempt = model_2.score(Success_Data[0][1])
    except:
        print"train %d model 2"%(i+1)
        return 0
        
    model_2_list.append(model_2)

    model_3 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=n_iteraton)
    model_3.startprob_ = start_prob
    #model_3.transmat_ = init_trans_mat
    model_3 = model_3.fit(train_Data[2],lengths=train_length[2])
    try:
        log_tempt = model_3.score(Success_Data[0][2])
    except:
        print"train %d model 3"%(i+1)
        return 0
        
    model_3_list.append(model_3)
    
    model_4 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                         params="mct", init_params="cmt", n_iter=n_iteraton)
    model_4.startprob_ = start_prob
    #model_4.transmat_ = init_trans_mat
    model_4 = model_4.fit(train_Data[3],lengths=train_length[3])
    try:
        log_tempt = model_4.score(Success_Data[0][3])
    except:
        print"train %d model 4"%(i+1)
        return 0
        
    model_4_list.append(model_4)


    #print "trial training progress (%d/%d) ...."%(i+1,success_train_num)
        

    # save the models
    if not os.path.isdir(model_save_path+"/multisequence_model"):
        os.makedirs(model_save_path+"/multisequence_model")
    
    joblib.dump(model_1, model_save_path+"/multisequence_model/model_s1.pkl")
    joblib.dump(model_2, model_save_path+"/multisequence_model/model_s2.pkl")
    joblib.dump(model_3, model_save_path+"/multisequence_model/model_s3.pkl")
    joblib.dump(model_4, model_save_path+"/multisequence_model/model_s4.pkl")
    joblib.dump(n_state, model_save_path+"/multisequence_model/n_state.pkl")
    joblib.dump(covariance_type_string, model_save_path+"/multisequence_model/covariance_type.pkl")
   
    #plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
