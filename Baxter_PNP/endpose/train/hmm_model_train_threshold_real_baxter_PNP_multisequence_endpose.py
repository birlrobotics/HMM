#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from hmmlearn.hmm import *
from sklearn.externals import joblib
import ipdb
from time import time
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
from sklearn.preprocessing import (
    scale,
    normalize
)



def matplot_list(list_data,
                 figure_index,
                 title,
                 label_string,
                 save_path,
                 save=False,
                 linewidth='3.0'):
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
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])*0.01).tolist()
        if label_string[i-1] == 'threshold':
            plt.plot(O, data, label=label_string[i-1],linewidth=1, linestyle = '--', mfc ="grey")
        else:
            plt.plot(O, data, label=label_string[i-1],linewidth=linewidth)
    plt.legend(loc='best', frameon=True)

    plt.title(title)

    #plt.annotate('State=4 Sub_State='+str(n_state)+' GaussianHMM_cov='+covariance_type_string,
    #         xy=(0, 0), xycoords='data',
    #         xytext=(+10, +30), textcoords='offset points', fontsize=16,
   #          arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

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

    df = df[[u'.endpoint_state.pose.position.x',
             u'.endpoint_state.pose.position.y',
             u'.endpoint_state.pose.position.z',
             u'.endpoint_state.pose.orientation.x',
             u'.endpoint_state.pose.orientation.y',
             u'.endpoint_state.pose.orientation.z',
             u'.endpoint_state.pose.orientation.w',
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



def array_list_mean(list_data,c,offset):
    """
    eg: argument list_data[X1,X2,...] X1,X2 numpy Column array 
        mean(X1,X2,..)  

        threshold = mean - c * std  

    return: marray [X_mean] , [X_std] [threshold]
    """
    tempt_list = []
    df_full_mean = pd.DataFrame()
    df_full_std = pd.DataFrame()
    df = pd.DataFrame()
    for data in list_data:
        df_tempt = pd.DataFrame(data=data)
        df = pd.concat([df,df_tempt], axis=1)
    mean_series = df.mean(axis=1)
    std_series = df.std(axis=1)

    return mean_series.values, std_series.values, (mean_series.values-std_series.values*c-offset)




def array_log_mean(list_data):
    tempt_list = []
    df = pd.DataFrame()
    for data in list_data:
        df_tempt = pd.DataFrame(data=data)
        df = pd.concat([df,df_tempt], axis=1)
    mean_series = df.mean(axis=1)
    std_series = df.std(axis=1)
    mean = mean_series.values.T.tolist()
    std = std_series.values.T.tolist()
    threshold = mean_series.values - 3*std_series.values
    return mean, std, threshold
        

    
def main():
    #ipdb.set_trace()


    preprocessing_scaling = False

    preprocessing_normalize = False

    data_feature = 7

    norm_style = 'l2'


    success_path = "/home/ben/ML_data/REAL_BAXTER_PICK_N_PLACE_6_1/success"

    model_save_path = "/home/ben/ML_data/REAL_BAXTER_PICK_N_PLACE_6_1/model/endpoint_pose"

    figure_save_path = "/home/ben/ML_data/REAL_BAXTER_PICK_N_PLACE_6_1/figure/endpoint_pose"

    success_train_num = 10

    threshold_constant = 10

    threshold_offset = 10

    n_state = joblib.load(model_save_path+'/multisequence_model/n_state.pkl')

    covariance_type_string = joblib.load(model_save_path+'/multisequence_model/covariance_type.pkl')

    global n_state
    global covariance_type_string


    
    #########--- load the success Data Index And Label String----###################
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

    
    ##########-----loading the Sucess trails data-----------############################
    ## Success_Data[trails][subtask][time_index,feature]
    ## Success_Index[trails][subtask]
    ## Success_Label_String[trails]
    for i in range(success_train_num):
        data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i],
                                            preprocessing_scaling=preprocessing_scaling,
                                            preprocessing_normalize=preprocessing_normalize,
                                            norm=norm_style)
        Success_Data.append(data_tempt)
        Success_Index.append(index_tempt)
        Success_Label_String.append("Success "+ path_index_name[i])



    
    
    ######-------loading the HMM Models to list[] ------#################################
    

    model_1 = joblib.load(model_save_path+"/multisequence_model/model_s1.pkl")
    model_2 = joblib.load(model_save_path+"/multisequence_model/model_s2.pkl")
    model_3 = joblib.load(model_save_path+"/multisequence_model/model_s3.pkl")
    model_4 = joblib.load(model_save_path+"/multisequence_model/model_s4.pkl")
        
#################################




    ###---------------get the minimun index in every subtask from trails-------#########
    Success_min_index = []
    index_tempt = []
    for j in range(4):
        index_tempt = []
        for i in range(success_train_num):
            index_tempt.append(Success_Index[i][j])
        Success_min_index.append(min(index_tempt))


    ######-----------training and getting the threshold ----################################
    model_1_log = []
    model_2_log = []
    model_3_log = []
    model_4_log = []
    model_1_log_T = []
    model_2_log_T = []
    model_3_log_T = []
    model_4_log_T = []
    model_1_end_mean = []
    model_2_end_mean = []
    model_3_end_mean = []
    model_4_end_mean = []
    # Training with left one out
    for i in range(success_train_num):
        #Do no train with its own model data
        model_1_log_tempt = []
        model_2_log_tempt = []
        model_3_log_tempt = []
        model_4_log_tempt = []
        for k in range(Success_min_index[0]):
            model_1_log_tempt.append(model_1.score(Success_Data[i][0][:k+1,:]))
    
              
        for k in range(Success_min_index[1]):
            model_2_log_tempt.append(model_2.score(Success_Data[i][1][:k+1,:]))


        for k in range(Success_min_index[2]):
            model_3_log_tempt.append(model_3.score(Success_Data[i][2][:k+1,:]))
        

        for k in range(Success_min_index[3]):
            model_4_log_tempt.append(model_4.score(Success_Data[i][3][:k+1,:]))

        model_1_log.append(np.asarray(model_1_log_tempt))
        model_2_log.append(np.asarray(model_2_log_tempt))
        model_3_log.append(np.asarray(model_3_log_tempt))
        model_4_log.append(np.asarray(model_4_log_tempt)) 



        print "computing the score log curves(%d/%d)"%(i+1,success_train_num)


    model_1_log_mean, model_1_log_std, model_1_threshold = array_list_mean(model_1_log,
                                                                           c = threshold_constant,
                                                                           offset=threshold_offset)
    print "computing the threshold curves 1"

    model_2_log_mean, model_2_log_std, model_2_threshold = array_list_mean(model_2_log,
                                                                           c = threshold_constant,
                                                                           offset=threshold_offset)
    print "computing the threshold curves 2"

    model_3_log_mean, model_3_log_std, model_3_threshold = array_list_mean(model_3_log,
                                                                           c = threshold_constant,
                                                                           offset=threshold_offset)
    print "computing the threshold curves 3"

    model_4_log_mean, model_4_log_std, model_4_threshold = array_list_mean(model_4_log,
                                                                           c = threshold_constant,
                                                                           offset=threshold_offset)

    print "computing the threshold curves 4"


    expected_log = [model_1_log_mean,
                    model_2_log_mean,
                    model_3_log_mean,
                    model_4_log_mean]

    threshold = [model_1_threshold,
                 model_2_threshold,
                 model_3_threshold,
                 model_4_threshold]

    if not os.path.isdir(figure_save_path+'/threshold'):
        os.makedirs(figure_save_path+'/threshold')

    matplot_list([expected_log[0],threshold[0]],
                 figure_index=10,
                 label_string=['expected_log','threshold'],
                 title='subtask 1 expected_log and threshold',
                 save=True,
                 save_path= figure_save_path+"/threshold")

    matplot_list([expected_log[1],threshold[1]],
                 figure_index=11,
                 label_string=['expected_log','threshold'],
                 title='subtask 2 expected_log and threshold',
                 save=True,
                 save_path=figure_save_path+'/threshold')

    matplot_list([expected_log[2],threshold[2]],
                 figure_index=12,
                 label_string=['expected_log','threshold'],
                 title='subtask 3 expected_log and threshold',
                 save=True,
                 save_path=figure_save_path+'/threshold')

    matplot_list([expected_log[3],threshold[3]],
                 figure_index=13,
                 label_string=['expected_log','threshold'],
                 title='subtask 4 expected_log and threshold',
                 save=True,
                 save_path=figure_save_path+'/threshold')

   # plt.show()

    if not os.path.isdir(model_save_path+"/multisequence_model"):
        os.makedirs(model_save_path+"/multisequence_model")
        
    joblib.dump(expected_log, model_save_path+"/multisequence_model/expected_log.pkl")
    joblib.dump(threshold, model_save_path+"/multisequence_model/threshold.pkl")
    joblib.dump(Success_min_index, model_save_path+"/multisequence_model/success_min_index.pkl")
    



    

    return 0

if __name__ == '__main__':
    sys.exit(main())
