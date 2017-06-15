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
                 linewidth='2.0',
                 fontsize= 50,
                 xaxis_interval=0.005):
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
    plt.xlabel('Time(s)',fontsize=fontsize)
  

    plt.ylabel('Log Liklihood',fontsize=fontsize)

    plt.xticks( fontsize = 50)
    plt.yticks( fontsize = 50)
    plt.ylim([-300000,200000])
    
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])*xaxis_interval).tolist()
        if label_string[i-1] == 'mean':
            plt.plot(O, data, label=label_string[i-1],linewidth='5.0', linestyle = '-', mfc ="black")
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
    


def load_data(path, path_index=1, preprocessing_scaling=False, preprocessing_normalize=False, norm='l2'):
    
    #df1 = pd.read_csv(path+'/R_Angles.dat', sep='\s+', header=None, skiprows=1)
    #df2 = pd.read_csv(path+'/R_CartPos.dat', sep='\s+', header=None, skiprows=1)
    df3 = pd.read_csv(path+'/R_Torques.dat', sep='\s+', header=None, skiprows=1)
    #df4 = pd.read_csv(path+'/worldforce-'+str(path_index)+".dat", sep='\s+', header=None, skiprows=1)
    df5 = pd.read_csv(path+'/R_State.dat', sep='\s+', header=None, skiprows=1)
    #df1.columns = ['time','s0','s1','s2','s3','s4','s5']
    #df2.columns = ['time','x','y','z','R','P','Y']
    df3.columns = ['time','Fx','Fy','Fz','Mx','My','Mz']
    #df4.columns = ['time','Fx','Fy','Fz','FR','FP','FY']

    df = df3
    #df = pd.merge(df1,df2, how='outer', on='time')
    #df = pd.merge(df,df3, how='outer', on='time')
    #df = pd.merge(df,df4, how='outer',on='time')
    df = df.fillna(method='ffill')

    df5.columns=['time']
    df5['state']=[1,2,3]
    df5.ix[3] = [0.005,0]
    df = pd.merge(df,df5, how='outer', on='time')
    df = df.fillna(method='ffill')
 
    X_1 = df.values[df.values[:,-1] ==0]
    index_1,column_1 = X_1.shape
    X_2 = df.values[df.values[:,-1] ==1]
    index_2,column_2 = X_2.shape
    X_3 = df.values[df.values[:,-1] ==2]
    index_3,column_3 = X_3.shape
    X_4 = df.values[df.values[:,-1] ==3]
    index_4,column_4 = X_4.shape
    
    index = [index_1,index_2,index_3,index_4]

    X_1_ = X_1[:,1:-1]
    X_2_ = X_2[:,1:-1]
    X_3_ = X_3[:,1:-1]
    X_4_ = X_4[:,1:-1]


    X_tempt = np.array([[0,0,0,0,0,0]])

    X_1_tempt = np.concatenate((X_tempt,X_1_),axis=0)
    X_1_tempt_1 = np.concatenate((X_1_,X_tempt),axis=0)
    X_1_d = X_1_tempt_1 - X_1_tempt
    X_1_d = X_1_d[1:-1]
    X_1_d = np.concatenate((X_tempt,X_1_d),axis=0)
    X_1_ = np.concatenate((X_1_,X_1_d),axis=1)

    X_2_tempt = np.concatenate((X_tempt,X_2_),axis=0)
    X_2_tempt_2 = np.concatenate((X_2_,X_tempt),axis=0)
    X_2_d = X_2_tempt_2 - X_2_tempt
    X_2_d = X_2_d[1:-1]
    X_2_d = np.concatenate((X_tempt,X_2_d),axis=0)
    X_2_ = np.concatenate((X_2_,X_2_d),axis=1)

    X_3_tempt = np.concatenate((X_tempt,X_3_),axis=0)
    X_3_tempt_3 = np.concatenate((X_3_,X_tempt),axis=0)
    X_3_d = X_3_tempt_3 - X_3_tempt
    X_3_d = X_3_d[1:-1]
    X_3_d = np.concatenate((X_tempt,X_3_d),axis=0)
    X_3_ = np.concatenate((X_3_,X_3_d),axis=1)

    X_4_tempt = np.concatenate((X_tempt,X_4_),axis=0)
    X_4_tempt_4 = np.concatenate((X_4_,X_tempt),axis=0)
    X_4_d = X_4_tempt_4 - X_4_tempt
    X_4_d = X_4_d[1:-1]
    X_4_d = np.concatenate((X_tempt,X_4_d),axis=0)
    X_4_ = np.concatenate((X_4_,X_4_d),axis=1)
    

    
    
    Data = [X_1_,X_2_,X_3_,X_4_]

    if preprocessing_normalize:
        normalize_X_1 = normalize(Data[0], norm=norm)
        normalize_X_2 = normalize(Data[1], norm=norm)
        normalize_X_3 = normalize(Data[2], norm=norm)
        normalize_X_4 = normalize(Data[3], norm=norm)
        
        index_1, column_1 = normalize_X_1.shape
        index_2, column_2 = normalize_X_1.shape
        index_3, column_3 = normalize_X_1.shape
        index_4, column_4 = normalize_X_1.shape

        Data = []
        Data = [normalize_X_1, normalize_X_2, normalize_X_3, normalize_X_4]
        index = [index_1, index_2, index_3, index_4]
        
    if preprocessing_scaling:
        scaled_X_1 = scale(Data[0])
        scaled_X_2 = scale(Data[1])
        scaled_X_3 = scale(Data[2])
        scaled_X_4 = scale(Data[3])

        index_1, column_1 = scaled_X_1.shape
        index_2, column_2 = scaled_X_1.shape
        index_3, column_3 = scaled_X_1.shape
        index_4, column_4 = scaled_X_1.shape

        Data = []
        Data = [scaled_X_1, scaled_X_2, scaled_X_3, scaled_X_4]
        
        index = [index_1,index_2,index_3,index_4]

    
        
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
        index, column = list_data[0].shape
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
    n_state = 3

    n_iteraton = 100

    covariance_type_string = 'diag'

    preprocessing_scaling = False

    preprocessing_normalize = False

    data_feature = 6

    norm_style = 'l2'

    success_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_SUCCESS"

    model_save_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_SUCCESS/model/wrench/30_trails"

    figure_save_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_SUCCESS/figure/wrench/30_trails"

    result_save_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_SUCCESS/result/wrench/30_trails"

    if not os.path.isdir(model_save_path+"/state_classification"):
        os.makedirs(model_save_path+"/state_classification")

    if not os.path.isdir(figure_save_path+"/state_classification"):
        os.makedirs(figure_save_path+"/state_classification")

    if not os.path.isdir(result_save_path+"/state_classification"):
        os.makedirs(result_save_path+"/state_classification")

    

    success_num = 30

    success_num_train = 29
    success_num_test = 1



    confusion_matrix = [[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]

    detection_time = [[],[],[],[]]


    
    # load the success Data Index And Label String
    path_index_name = []
    for i in range(1,45):
        if i+1 <= 9:
            post_str = '0'+str(i+1)
        else:
            post_str = str(i+1)
        path_index_name.append('20121127-HIROSA-S-'+post_str)


    for o in range(success_num):
        Success_Data_Train = []
        Success_Index_Train = []
        Success_Label_String_Train = []

        Success_Data_Test = []
        Success_Index_Test = []
        Success_Label_String_Test = []
        print "Choosing CV index"
        for i in range(success_num):
            if not i==o:
                data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i],
                                                    preprocessing_scaling=preprocessing_scaling,
                                                    preprocessing_normalize=preprocessing_normalize,
                                                    norm=norm_style)
                Success_Data_Train.append(data_tempt)
                Success_Index_Train.append(index_tempt)
                Success_Label_String_Train.append(path_index_name[i])
                print"%d " %i

            else:
                data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i])
                Success_Data_Test.append(data_tempt)
                Success_Index_Test.append(index_tempt)
                Success_Label_String_Test.append(path_index_name[i]) 




        # get the FX data list in State 2 and get the data mean and std
        ##  Success Data [Trails] [Subtask] [index,columns]
        # success_data_fx_list = []
        # success_data_fx_string_list = Success_Label_String
        # for i in range(success_trail_num):
        #     success_data_fx_list.append(Success_Data[i][1][:,0])

        # success_S2_mean_data, success_S2_std_data = array_list_mean(success_data_fx_list)

        # success_data_fx_list = []
        # for i in range(success_trail_num):
        #     success_data_fx_list.append(Success_Data[i][1][:,0].T)

        # success_data_fx_list.append(success_S2_mean_data.T[0,:])
        # success_data_fx_string_list.append("mean")




        # matplot_list(success_data_fx_list,
        #              figure_index=10,
        #              title="Rotation R_Torques data Fx and Mean in Success trails ",
        #              save=False,
        #              label_string=success_data_fx_string_list,
        #              save_path = model_save_path+"/train_model/figure/")

        #plt.show()


        #averaging the data
        train_Data = []
        train_length = []
        train_length_tempt = []

        for i in range(success_num_train):
            if i==0:
                data_tempt = Success_Data_Train[i][0]
                train_length_tempt = []
            else:
                data_tempt = np.concatenate((data_tempt,Success_Data_Train[i][0]),axis = 0)
            train_length_tempt.append(Success_Index_Train[i][0])
        train_Data.append(data_tempt)
        train_length.append(train_length_tempt)


        for i in range(success_num_train):
            if i==0:
                data_tempt = Success_Data_Train[i][1]
                train_length_tempt = []
            else:
                data_tempt = np.concatenate((data_tempt,Success_Data_Train[i][1]),axis = 0)
            train_length_tempt.append(Success_Index_Train[i][1])
        train_Data.append(data_tempt)
        train_length.append(train_length_tempt)


        for i in range(success_num_train):
            if i==0:
                data_tempt = Success_Data_Train[i][2]
                train_length_tempt = []
            else:
                data_tempt = np.concatenate((data_tempt,Success_Data_Train[i][2]),axis = 0)
            train_length_tempt.append(Success_Index_Train[i][2])
        train_Data.append(data_tempt)
        train_length.append(train_length_tempt)


        for i in range(success_num_train):
            if i==0:
                data_tempt = Success_Data_Train[i][3]
                train_length_tempt = []
            else:
                data_tempt = np.concatenate((data_tempt,Success_Data_Train[i][3]),axis = 0)
            train_length_tempt.append(Success_Index_Train[i][3])
        train_Data.append(data_tempt)
        train_length.append(train_length_tempt)
        




        #plt_data = [train_Data[0][:,2].T,train_Data[1][:,2].T,train_Data[2][:,2].T,train_Data[3][:,2].T]
        
        # matplot_list(plt_data,
        #              figure_index=10,
        #              title="Rotation R_Torques data Fx and Mean in Success trails ",
        #              save=False,
        #              label_string=['A','R','I','M'],
        #              save_path = model_save_path+"/train_model/figure/")
        #plt.show()



        #initial start probalility    
        start_prob = np.zeros(n_state)

        start_prob[0] = 1
        # Subtasks i hmm model list
        model_1_list = []
        model_2_list = []
        model_3_list = []
        model_4_list = []



        model_1 =GaussianHMM(n_components=n_state, covariance_type=covariance_type_string,
                             params="mct", init_params="cmt", n_iter=n_iteraton)
        model_1.startprob_ = start_prob
        #model_1.transmat_ = init_trans_mat
        model_1 = model_1.fit(train_Data[0],lengths=train_length[0])
        try:
            log_tempt = model_1.score(Success_Data_Train[0][0])
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
            log_tempt = model_2.score(Success_Data_Train[0][1])
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
            log_tempt = model_3.score(Success_Data_Train[0][2])
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
            log_tempt = model_4.score(Success_Data_Train[0][3])
        except:
            print"train %d model 4"%(i+1)
            return 0

        model_4_list.append(model_4)

        



        # joblib.dump(model_1, model_save_path+'/train_model/my_best_model/model_s1.pkl')
        # joblib.dump(model_2, model_save_path+'/train_model/my_best_model/model_s2.pkl')
        # joblib.dump(model_3, model_save_path+'/train_model/my_best_model/model_s3.pkl')
        # joblib.dump(model_4, model_save_path+'/train_model/my_best_model/model_s4.pkl')

        # best_model = [model_1,
        #               model_2,
        #               model_3,
        #               model_4]
      

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
        model_1_log_full = []
        model_2_log_full = []
        model_3_log_full = []
        model_4_log_full = []

        model_log_full = []
        # Training with left one out

        for i in range(success_num_test):
            #Do no train with its own model data
            model_1_log = np.array([0])
            model_2_log = np.array([0])
            model_3_log = np.array([0])
            model_4_log = np.array([0])
            cul_time = 0
            Classification_Full_Flag = True
            for j in range(4):
                model_1_log_tempt = []
                model_2_log_tempt = []
                model_3_log_tempt = []
                model_4_log_tempt = []
                classification_flag = False
                for k in range(Success_Index_Test[i][j]):
                    data_1 = model_1.score(Success_Data_Test[i][j][:k+1,:])
                    data_2 = model_2.score(Success_Data_Test[i][j][:k+1,:])
                    data_3 = model_3.score(Success_Data_Test[i][j][:k+1,:])
                    data_4 = model_4.score(Success_Data_Test[i][j][:k+1,:])


                    model_1_log_tempt.append(data_1)
                    model_2_log_tempt.append(data_2)
                    model_3_log_tempt.append(data_3)
                    model_4_log_tempt.append(data_4)

                    arg_list = np.argsort(np.asarray([data_1, data_2, data_3, data_4]))

                    if not arg_list[-1] == j:
                        classification_flag = False
                    elif not classification_flag:
                        classification_flag =True
                        latest_cross_time = k

                if not classification_flag:
                    print"False classification at state %d" %(j+1)
                    print"Misclassifcation to state %d"%(arg_list[-1]+1)
                    Classification_Full_Flag = False
                else:
                    print "the cross time is %f" %((cul_time+latest_cross_time)*0.005)
                    print "the time of dectection is %f" %(latest_cross_time*100/Success_Index_Test[i][j]) + "%"
                confusion_matrix[j][arg_list[-1]] = confusion_matrix[j][arg_list[-1]]+1
                detection_time[j].append(latest_cross_time*100/Success_Index_Test[i][j])





                model_1_log = np.concatenate((model_1_log, np.asarray(model_1_log_tempt)))
                model_2_log = np.concatenate((model_2_log, np.asarray(model_2_log_tempt)))
                model_3_log = np.concatenate((model_3_log, np.asarray(model_3_log_tempt)))
                model_4_log = np.concatenate((model_4_log, np.asarray(model_4_log_tempt)))
                cul_time = Success_Index_Test[i][j] +cul_time


            print "confusion_matrix ="
            print "%d %d %d %d" %(confusion_matrix[0][0],confusion_matrix[0][1],confusion_matrix[0][2],confusion_matrix[0][3])
            print "%d %d %d %d" %(confusion_matrix[1][0],confusion_matrix[1][1],confusion_matrix[1][2],confusion_matrix[1][3])
            print "%d %d %d %d" %(confusion_matrix[2][0],confusion_matrix[2][1],confusion_matrix[2][2],confusion_matrix[2][3])
            print "%d %d %d %d" %(confusion_matrix[3][0],confusion_matrix[3][1],confusion_matrix[3][2],confusion_matrix[3][3])
            print "computing the classification log curves of testing (%d/%d)"%(i+1,success_num_test)
            print "Detection Time: %d %d %d %d" %(detection_time[0][-1],detection_time[1][-1],detection_time[2][-1],detection_time[3][-1])

        
            if Classification_Full_Flag:
                matplot_list([model_1_log,model_2_log,model_3_log,model_4_log],
                             figure_index=o,
                             label_string=['Approach','Rotation','Insertion','Mating'],
                             title='State Classification'+Success_Label_String_Test[i],
                             save=True,
                             save_path = figure_save_path+"/state_classification")
            else:
                matplot_list([model_1_log,model_2_log,model_3_log,model_4_log],
                             figure_index=o,
                             label_string=['Approach','Rotation','Insertion','Mating'],
                             title='Wrong State Classification'+Success_Label_String_Test[i],
                             save=True,
                             save_path = figure_save_path+"/state_classification")


                
            np.savetxt(result_save_path+'/state_classification/confusion_matrix.dat', np.asarray(confusion_matrix), fmt='%d')
            np.savetxt(result_save_path+'/state_classification/detection_time.dat', np.asarray(detection_time), fmt='%d')
            average_time = [np.mean(detection_time[0]),
                            np.mean(detection_time[1]),
                            np.mean(detection_time[2]),
                            np.mean(detection_time[3])]
            np.savetxt(result_save_path+'/state_classification/average_detection_time.dat', average_time, fmt='%d')
            


            #plt.show()


    
    #plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
