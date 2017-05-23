#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from hmmlearn.hmm import *
from sklearn.externals import joblib
import ipdb
import time
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

    plt.annotate('State=4 Sub_State='+str(n_state)+' GaussianHMM_cov='+covariance_type_string,
             xy=(0, 0), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    if save:
        plt.savefig(save_path+title+".eps", format="eps")




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
    
   # df1 = pd.read_csv(path+'/R_Angles.dat', sep='\s+', header=None, skiprows=1)
   # df2 = pd.read_csv(path+'/R_CartPos.dat', sep='\s+', header=None, skiprows=1)
    df3 = pd.read_csv(path+'/R_Torques.dat', sep='\s+', header=None, skiprows=1)
    #df4 = pd.read_csv(path+'/worldforce-'+str(path_index)+".dat", sep='\s+', header=None, skiprows=1)
    df5 = pd.read_csv(path+'/R_State.dat', sep='\s+', header=None, skiprows=1)

   # df1.columns = ['time','s0','s1','s2','s3','s4','s5']
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

    TPR_list_tempt = []
    FPR_list_tempt = []

    report_list = []

    TPR_list = []
    FPR_list = []
    


    c_list = [0,1,2,3,4]
    for c in c_list:
        for p in range(2):

            TPR_list_tempt = []
            FPR_list_tempt = []

            #ipdb.set_trace()

            global n_state
            global covariance_type_string

            n_iteraton = 100

            covariance_type_string = 'full'

            preprocessing_scaling = False

            preprocessing_normalize = False

            data_feature = 6

            norm_style = 'l2'

            success_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_SUCCESS"

            model_save_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_SUCCESS/train_model"

            success_train_trails = 22

            threshold_constant = c

            threshold_offset = 10

            n_state = joblib.load(model_save_path+'/model_decision/n_state.pkl')


            CV_Fold_Success_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 33,34,35,36,37,38,39,40,41,42,43]
            CV_Fold_Success_2 = [11,12,13,14,15,16,17,18,19,20,21,
                                 22,23,24,25,26,27,28,29,30,31,32]
            
            if p == 0:
                CV_Success_Train = CV_Fold_Success_1
                CV_Success_Test = CV_Fold_Success_2
            else:
                CV_Success_Train = CV_Fold_Success_2
                CV_Success_Test = CV_Fold_Success_1

            ##CV!!!!!!!!!


            #CV_index_choose = int(raw_input("Please choose CV Variable 0~3 \n"))


            #########--- load the success Data Index And Label String----###################
            path_index_name = []

            for i in range(1,45):
                if i+1 <= 9:
                    post_str = '0'+str(i+1)
                else:
                    post_str = str(i+1)

                path_index_name.append('20121127-HIROSA-S-'+post_str)


            Success_Data_Train = []
            Success_Index_Train = []
            Success_Label_String_Train = []


            ##########-----loading the Sucess trails data-----------############################
            ## Success_Data[trails][subtask][time_index,feature]
            ## Success_Index[trails][subtask]
            ## Success_Label_String[trails]
            print "Choosing CV index"
            for i in range(44):
                if i in CV_Success_Train:
                    data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i],
                                                        preprocessing_scaling=preprocessing_scaling,
                                                        preprocessing_normalize=preprocessing_normalize,
                                                        norm=norm_style)
                    Success_Data_Train.append(data_tempt)
                    Success_Index_Train.append(index_tempt)
                    Success_Label_String_Train.append(path_index_name[i])
                    print"%d " %i


            ######################################################################################


            ######-------loading the HMM Models to list[] ------#################################

            ## Subtasks i hmm model list
            ## model_i[trails]  
            model_1_list = []
            model_2_list = []
            model_3_list = []
            model_4_list = []


            # loading the models
            for i in range(success_train_trails):
                if not os.path.isdir(success_path+'/'+Success_Label_String_Train[i]):
                    print success_path+'/'+Success_Label_String_Train[i]+" do not exit"
                    print "Please train the hmm model first, and check your model folder"

                model_1_list.append(joblib.load(model_save_path+'/'+Success_Label_String_Train[i]+"/model_s1.pkl"))
                model_2_list.append(joblib.load(model_save_path+'/'+Success_Label_String_Train[i]+"/model_s2.pkl"))
                model_3_list.append(joblib.load(model_save_path+'/'+Success_Label_String_Train[i]+"/model_s3.pkl"))
                model_4_list.append(joblib.load(model_save_path+'/'+Success_Label_String_Train[i]+"/model_s4.pkl"))

        ###########train the process mornitoring, subtasks models log likelihood curves in one trail 1
       
            ###############################################################################


            ###---------------get the minimun index in every subtask from trails-------#########
            Success_min_index_train = []
            index_tempt = []
            for j in range(4):
                index_tempt = []
                for i in range(success_train_trails):
                    index_tempt.append(Success_Index_Train[i][j])
                    Success_min_index_train.append(min(index_tempt))


            ######-----------training and getting the threshold ----################################
            # Training with left n-1 out
            model_1_end_mean = []
            model_2_end_mean = []
            model_3_end_mean = []
            model_4_end_mean = []
            for n in range(success_train_trails):
                model_1_log = []
                model_2_log = []
                model_3_log = []
                model_4_log = []
                for i in range(success_train_trails):
                    #Do no train with its own model data
                    if not i==n:
                        model_1_log_tempt = []
                        model_2_log_tempt = []
                        model_3_log_tempt = []
                        model_4_log_tempt = []
                        for j in range(4):
                            if j == 0:
                                try:
                                    model_1_log_tempt = model_1_list[n].score(Success_Data_Train[i][j][:,:])
                                    model_1_log.append(model_1_log_tempt)
                                except:
                                    print "error in trail %d model 1"%(n+1)
                                    return 0

                            elif j == 1:
                                try:
                                    model_2_log_tempt = model_2_list[n].score(Success_Data_Train[i][j][:,:])
                                    model_2_log.append(model_2_log_tempt)
                                except:
                                    print "error in trail %d model 2"%(n+1)
                                    return 0
                            elif j==2:
                                try:
                                    model_3_log_tempt = model_3_list[n].score(Success_Data_Train[i][j][:,:])
                                    model_3_log.append(model_3_log_tempt)
                                except:
                                    print "error in trail %d model 3"%(n+1)
                                    return 0        
                            elif j==3:
                                try:
                                    model_4_log_tempt = model_4_list[n].score(Success_Data_Train[i][j][:,:])
                                    model_4_log.append(model_4_log_tempt)
                                except:
                                    print "error in trail %d model 4"%(n+1)
                                    return 0

                mean_tempt = np.mean(np.asarray(model_1_log))
                model_1_end_mean.append(mean_tempt)
                mean_tempt = np.mean(np.asarray(model_2_log))
                model_2_end_mean.append(mean_tempt)
                mean_tempt = np.mean(np.asarray(model_3_log))
                model_3_end_mean.append(mean_tempt)
                mean_tempt = np.mean(np.asarray(model_4_log))
                model_4_end_mean.append(mean_tempt)





            # get the best model according to the highest log-likelihood mean value
            argsort_model_1_end_mean = np.argsort(np.asarray(model_1_end_mean))
            argsort_model_2_end_mean = np.argsort(np.asarray(model_2_end_mean))
            argsort_model_3_end_mean = np.argsort(np.asarray(model_3_end_mean))
            argsort_model_4_end_mean = np.argsort(np.asarray(model_4_end_mean))
            best_model_trail_list = [argsort_model_1_end_mean[-1],
                                     argsort_model_2_end_mean[-1],
                                     argsort_model_3_end_mean[-1],
                                     argsort_model_4_end_mean[-1]]
            print "the best model_1 is %d"%(argsort_model_1_end_mean[-1]+1)
            print "the best model_1_mean is %d"%model_1_end_mean[argsort_model_1_end_mean[-1]]
            print "the best model_2 is %d"%(argsort_model_2_end_mean[-1]+1)
            print "the best model_2_mean is %d"%model_2_end_mean[argsort_model_2_end_mean[-1]]
            print "the best model_3 is %d"%(argsort_model_3_end_mean[-1]+1)
            print "the best model_3_mean is %d"%model_3_end_mean[argsort_model_3_end_mean[-1]]
            print "the best model_4 is %d"%(argsort_model_4_end_mean[-1]+1)
            print "the best model_4_mean is %d"%model_4_end_mean[argsort_model_4_end_mean[-1]]



            best_model = [model_1_list[argsort_model_1_end_mean[-1]],
                          model_2_list[argsort_model_2_end_mean[-1]],
                          model_3_list[argsort_model_3_end_mean[-1]],
                          model_4_list[argsort_model_4_end_mean[-1]]]
            best_model_arg = [argsort_model_1_end_mean[-1],
                              argsort_model_2_end_mean[-1],
                              argsort_model_3_end_mean[-1],
                              argsort_model_4_end_mean[-1]]




            ###---------------get the minimun index in every subtask from trails-------#########
            Success_min_index_train = []
            index_tempt = []
            for j in range(4):
                index_tempt = []
                for i in range(success_train_trails):
                    index_tempt.append(Success_Index_Train[i][j])
                    Success_min_index_train.append(min(index_tempt))


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
            for i in range(success_train_trails):
                #Do no train with its own model data
                model_1_log_tempt = []
                model_2_log_tempt = []
                model_3_log_tempt = []
                model_4_log_tempt = []
                if not i == best_model_arg[0]:
                    for k in range(Success_min_index_train[0]):
                        model_1_log_tempt.append(best_model[0].score(Success_Data_Train[i][0][:k+1,:]))


                if not i == best_model_arg[1]:
                    for k in range(Success_min_index_train[1]):
                        model_2_log_tempt.append(best_model[1].score(Success_Data_Train[i][1][:k+1,:]))


                if not i == best_model_arg[2]:
                    for k in range(Success_min_index_train[2]):
                        model_3_log_tempt.append(best_model[2].score(Success_Data_Train[i][2][:k+1,:]))


                if not i == best_model_arg[3]:
                    for k in range(Success_min_index_train[3]):
                        model_4_log_tempt.append(best_model[3].score(Success_Data_Train[i][3][:k+1,:]))

                model_1_log.append(np.asarray(model_1_log_tempt))
                model_2_log.append(np.asarray(model_2_log_tempt))
                model_3_log.append(np.asarray(model_3_log_tempt))
                model_4_log.append(np.asarray(model_4_log_tempt)) 



                print "computing the score log curves(%d/%d)"%(i+1,success_train_trails)


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

            matplot_list([expected_log[0],threshold[0]],
                         figure_index=10,
                         label_string=['expected_log','threshold'],
                         title='subtask 1 expected_log and threshold',
                         save=True,
                         save_path= model_save_path+"/figure/")

            matplot_list([expected_log[1],threshold[1]],
                         figure_index=11,
                         label_string=['expected_log','threshold'],
                         title='subtask 2 expected_log and threshold',
                         save=True,
                         save_path= model_save_path+"/figure/")

            matplot_list([expected_log[2],threshold[2]],
                         figure_index=12,
                         label_string=['expected_log','threshold'],
                         title='subtask 3 expected_log and threshold',
                         save=True,
                         save_path= model_save_path+"/figure/")

            matplot_list([expected_log[3],threshold[3]],
                         figure_index=13,
                         label_string=['expected_log','threshold'],
                         title='subtask 4 expected_log and threshold',
                         save=True,
                         save_path= model_save_path+"/figure/")

            #plt.show()

            if not os.path.isdir(model_save_path+"/model_decision/"):
                os.makedirs(model_save_path+"/model_decision/")

            joblib.dump(best_model, model_save_path+"/model_decision/best_model.pkl")
            joblib.dump(expected_log, model_save_path+"/model_decision/expected_log.pkl")
            joblib.dump(threshold, model_save_path+"/model_decision/threshold.pkl")
            joblib.dump(Success_min_index_train, model_save_path+"/model_decision/success_min_index.pkl")




            for o in range(2):


                success_test_trails = 22

                fail_test_trails = 8 

                #[Success_index, Fail_index]

                CV_Fold_Fail_1 = [0, 1, 2, 3, 4, 5, 6, 7]
                CV_Fold_Fail_2 = [8, 9,10,11,12,13,14,15]

                if o ==0:
                    CV_Fail_Test = CV_Fold_Fail_1
                else:
                    CV_Fail_Test = CV_Fold_Fail_2
                

                n_state = joblib.load(model_save_path+'/model_decision/n_state.pkl')

                
                #0~19
                #CV_index_choose = int(raw_input("Please choose CV Variable 0~3 \n"))


                #########--- load the success Data Index And Label String----###################

                path_index_name = []
                for i in range(1,45):
                    if i+1 <= 9:
                        post_str = '0'+str(i+1)
                    else:
                        post_str = str(i+1)
                    path_index_name.append('20121127-HIROSA-S-'+post_str)

                Success_Data_Test = []
                Success_Index_Test = []
                Success_Label_String_Test = []

                ##########-----loading the Sucess trails data-----------############################
                ## Success_Data[trails][subtask][time_index,feature]
                ## Success_Index[trails][subtask]
                ## Success_Label_String[trails]
                print "choosing success CV index"
                for i in range(44):
                    if i in CV_Success_Test:
                        data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i])
                        Success_Data_Test.append(data_tempt)
                        Success_Index_Test.append(index_tempt)
                        Success_Label_String_Test.append(path_index_name[i])
                        print"%d" %i






                fail_path = "/home/ben/ML_data/REAL_HIRO_ONE_SA_ERROR_CHARAC"
                fail_path_index_name = []

                for i in range(1,17):
                    if i+1 <= 9:
                        post_str = '0'+str(i+1)
                    else:
                        post_str = str(i+1)
                    fail_path_index_name.append('20160930-HIRO_ERROR-'+post_str)

                    
                Fail_Data_Test = []
                Fail_Index_Test = []
                Fail_Label_String_Test = []

                ##########-----loading the Sucess trails data-----------############################
                ## Success_Data[trails][subtask][time_index,feature]
                ## Success_Index[trails][subtask]
                ## Success_Label_String[trails]
                print "choosing CV fail index"
                for i in range(16):
                    if i in CV_Fail_Test:
                        data_tempt, index_tempt = load_data(path=fail_path+"/"+fail_path_index_name[i])
                        Fail_Data_Test.append(data_tempt)
                        Fail_Index_Test.append(index_tempt)
                        Fail_Label_String_Test.append(fail_path_index_name[i])
                        print"%d" %i


              ######-------loading the HMM Models to list[] ------#################################

              ## Subtasks i hmm model list
              ## model_i[trails]  
                model_1_list = []
                model_2_list = []
                model_3_list = []
                model_4_list = []


                # loading the models

                best_model = joblib.load(model_save_path+"/model_decision/best_model.pkl")
                expected_log = joblib.load(model_save_path+"/model_decision/expected_log.pkl")
                threshold = joblib.load(model_save_path+"/model_decision/threshold.pkl")
                Success_min_index = joblib.load(model_save_path+"/model_decision/success_min_index.pkl")             




                if o == 0:

                  ######-----------testing and getting the threshold ----################################

                    model_1_log = []
                    model_2_log = []
                    model_3_log = []
                    model_4_log = []
                    # Testing with left one out
                    for i in range(success_test_trails):
                        #Do no test with its own model data
                        model_1_log_tempt = []
                        model_2_log_tempt = []
                        model_3_log_tempt = []
                        model_4_log_tempt = []
                        for k in range(min([Success_min_index_train[0],Success_Index_Test[i][0]])):
                            model_1_log_tempt.append(best_model[0].score(Success_Data_Test[i][0][:k+1,:]))


                        for k in range(min([Success_min_index_train[1],Success_Index_Test[i][1]])):
                            model_2_log_tempt.append(best_model[1].score(Success_Data_Test[i][1][:k+1,:]))


                        for k in range(min([Success_min_index_train[2],Success_Index_Test[i][2]])):
                            model_3_log_tempt.append(best_model[2].score(Success_Data_Test[i][2][:k+1,:]))


                        for k in range(min([Success_min_index_train[3],Success_Index_Test[i][3]])):
                            model_4_log_tempt.append(best_model[3].score(Success_Data_Test[i][3][:k+1,:]))

                        model_1_log.append(np.asarray(model_1_log_tempt))
                        model_2_log.append(np.asarray(model_2_log_tempt))
                        model_3_log.append(np.asarray(model_3_log_tempt))
                        model_4_log.append(np.asarray(model_4_log_tempt)) 



                        print "computing the test data score log curves(%d/%d)"%(i+1,success_test_trails)


                    test_log_full = []
                    test_log_tempt = []

                    for j in range(success_test_trails):
                        test_log_tempt = model_1_log[j]
                        test_log_tempt = np.concatenate((test_log_tempt, model_2_log[j]), axis =0)
                        test_log_tempt = np.concatenate((test_log_tempt, model_3_log[j]), axis =0)
                        test_log_tempt = np.concatenate((test_log_tempt, model_4_log[j]), axis =0)
                        test_log_full.append(test_log_tempt)

                    threshold_full = threshold[0][:Success_min_index[0]]

                    threshold_full = np.concatenate((threshold_full,threshold[1][:Success_min_index[1]]), axis = 0)
                    threshold_full = np.concatenate((threshold_full,threshold[2][:Success_min_index[2]]), axis = 0)
                    threshold_full = np.concatenate((threshold_full,threshold[3][:Success_min_index[3]]), axis = 0)


                    test_diff_log_full = []
                    test_diff_log_tempt = []
                    TP = 0
                    FN = 0
                    for j in range(success_test_trails):
                        test_diff_log_tempt = model_1_log[j] - threshold[0][:min([Success_min_index_train[0],Success_Index_Test[j][0]])]
                        test_diff_log_tempt = np.concatenate((test_diff_log_tempt, model_2_log[j]-threshold[1][:min([Success_min_index_train[1],Success_Index_Test[j][1]])]), axis =0)
                        test_diff_log_tempt = np.concatenate((test_diff_log_tempt, model_3_log[j]-threshold[2][:min([Success_min_index_train[2],Success_Index_Test[j][2]])]), axis =0)
                        test_diff_log_tempt = np.concatenate((test_diff_log_tempt, model_4_log[j]-threshold[3][:min([Success_min_index_train[3],Success_Index_Test[j][3]])]), axis =0)
                        test_diff_log_full.append(test_diff_log_tempt)

                        flag_tempt = True
                        n = 0
                        for data in test_diff_log_tempt:
                            n = n+1
                            if data < 0:
                                FN = FN + 1
                                flag_tempt = False
                                print "FN at test No %d" %(j+1)
                                print "Fail at index %f seconds" %(n*0.005)
                                break
                        if flag_tempt:
                            TP = TP + 1
                            print "TP at test No %d" %(j+1)
                            print "TP = %d , FN = %d"%(TP,FN)

                    subtask_name_list = ['Approach',
                                         'Rotation',
                                         'Insertion',
                                         'Mantain']

                    if not os.path.isdir(model_save_path+"/figure/"):
                        os.makedirs(model_save_path+"/figure/")
                    for j in range(success_test_trails):
                        matplot_list(list_data=[test_log_full[j],threshold_full],
                                     figure_index=j,
                                     label_string=['current log','threshold'],
                                     title=' current log and threshold in Test'+Success_Label_String_Test[j],
                                     save=True,
                                     save_path = model_save_path+"/figure/")

                    print "True Positive Rate(TPR) = %f" %(TP*100/(TP+FN)) +"%"


                    TPR = TP*100/(TP+FN)

                    TPR_list_tempt.append(TPR)










               ######-------loading the HMM Models to list[] ------#################################

               ## Subtasks i hmm model list
               ## model_i[trails]  





               ######-----------testing and getting the threshold ----################################

                model_1_log = []
                model_2_log = []
                model_3_log = []
                model_4_log = []
                # Testing with left one out
                for i in range(fail_test_trails):
                    #Do no test with its own model data
                    model_1_log_tempt = []
                    model_2_log_tempt = []
                    model_3_log_tempt = []
                    model_4_log_tempt = []
                    for k in range(Fail_Index_Test[i][0]):
                        model_1_log_tempt.append(best_model[0].score(Fail_Data_Test[i][0][:k+1,:]))


                    for k in range(Fail_Index_Test[i][1]):
                        model_2_log_tempt.append(best_model[1].score(Fail_Data_Test[i][1][:k+1,:]))


                    for k in range(Fail_Index_Test[i][2]):
                        model_3_log_tempt.append(best_model[2].score(Fail_Data_Test[i][2][:k+1,:]))


                    for k in range(Fail_Index_Test[i][3]):
                        model_4_log_tempt.append(best_model[3].score(Fail_Data_Test[i][3][:k+1,:]))

                    model_1_log.append(np.asarray(model_1_log_tempt))
                    model_2_log.append(np.asarray(model_2_log_tempt))
                    model_3_log.append(np.asarray(model_3_log_tempt))
                    model_4_log.append(np.asarray(model_4_log_tempt)) 



                    print "computing the test failure  data score log curves(%d/%d)"%(i+1,fail_test_trails)


                test_log_full = []
                test_log_tempt = []

                FP = 0
                TN = 0

                fail_min_index = []
                fail_min_index_tempt = []

                for j in range(fail_test_trails):
                    fail_min_index_tempt = []
                    if Success_min_index_train[0]>Fail_Index_Test[j][0]:
                        fail_min_index_tempt.append(Fail_Index_Test[j][0])
                    else:
                        fail_min_index_tempt.append(Success_min_index_train[0])

                    if Success_min_index_train[1]>Fail_Index_Test[j][1]:
                        fail_min_index_tempt.append(Fail_Index_Test[j][1])
                    else:
                        fail_min_index_tempt.append(Success_min_index_train[1])

                    if Success_min_index_train[2]>Fail_Index_Test[j][2]:
                        fail_min_index_tempt.append(Fail_Index_Test[j][2])
                    else:
                        fail_min_index_tempt.append(Success_min_index_train[2])

                    if Success_min_index_train[3]>Fail_Index_Test[j][3]:
                        fail_min_index_tempt.append(Fail_Index_Test[j][3])
                    else:
                        fail_min_index_tempt.append(Success_min_index_train[3])

                    fail_min_index.append(fail_min_index_tempt)


                fail_log_full = []

                test_diff_log_full = []
                test_diff_log_tempt = []

                for j in range(fail_test_trails):

                    test_diff_log_tempt = model_1_log[j][:fail_min_index[j][0]]-threshold[0][:fail_min_index[j][0]]
                    test_diff_log_tempt = np.concatenate((test_diff_log_tempt, model_2_log[j][:fail_min_index[j][1]]-threshold[1][:fail_min_index[j][1]]), axis =0)
                    test_diff_log_tempt = np.concatenate((test_diff_log_tempt, model_3_log[j][:fail_min_index[j][2]]-threshold[2][:fail_min_index[j][2]]), axis =0)
                    test_diff_log_tempt = np.concatenate((test_diff_log_tempt, model_4_log[j][:fail_min_index[j][3]]-threshold[3][:fail_min_index[j][3]]), axis =0)

                    test_diff_log_full.append(test_diff_log_tempt)

                    fail_log_full_tempt = model_1_log[j][:fail_min_index[j][0]]
                    fail_log_full_tempt = np.concatenate(( fail_log_full_tempt, model_2_log[j][:fail_min_index[j][1]]))
                    fail_log_full_tempt = np.concatenate(( fail_log_full_tempt, model_3_log[j][:fail_min_index[j][2]]))
                    fail_log_full_tempt = np.concatenate(( fail_log_full_tempt, model_4_log[j][:fail_min_index[j][3]]))
                    fail_log_full.append(fail_log_full_tempt)





                    flag_tempt = True
                    n = 0
                    for data in test_diff_log_tempt:
                        n = n+1
                        if data < 0:
                            TN = TN + 1
                            flag_tempt = False
                            print "TN at test No %d" %(j+1)
                            print "Fail at index %f seconds" %(n*0.005)
                            break

                    if flag_tempt:
                        FP = FP + 1
                        print "FP at test No %d" %(j+1)



                print "FP = %d , TN = %d"%(FP,TN)

                subtask_name_list = ['Approach',
                                     'Rotation',
                                     'Insertion',
                                     'Mantain']


                if not os.path.isdir(model_save_path+"/figure/"):
                    os.makedirs(model_save_path+"/figure/")

                for j in range(fail_test_trails):


                    fail_threshold = threshold[0][:fail_min_index[j][0]]
                    fail_threshold = np.concatenate((fail_threshold, threshold[1][:fail_min_index[j][1]]))
                    fail_threshold = np.concatenate((fail_threshold, threshold[2][:fail_min_index[j][2]]))
                    fail_threshold = np.concatenate((fail_threshold, threshold[3][:fail_min_index[j][3]]))

                    matplot_list(list_data=[fail_log_full[j],fail_threshold],
                                 figure_index=j+40,
                                 label_string=['current log','threshold'],
                                 title=' current log and threshold in Test'+Fail_Label_String_Test[j],
                                 save=True,
                                 save_path = model_save_path+"/figure/")


                print "True Positive Rate(TPR) = %f" %(TP*100/(TP+FN)) +"%"
                print "False Positive Rate(FPR) = %f" %(FP*100/(FP+TN)) +"%"


                TPR = TP*100/(TP+FN)

                FPR = FP*100/(FP+TN)

                TPR_list_tempt.append(TPR)
                FPR_list_tempt.append(FPR)

                print"Computing TPR FPR one time finish!!!"
                print"%d" %(np.array(TPR_list_tempt).shape)




        TPR_list.append(np.average(TPR_list_tempt))
        FPR_list.append(np.average(FPR_list_tempt))

        report_list.append([c, np.average(TPR_list_tempt),np.average(FPR_list_tempt)])

        np.savetxt(model_save_path+'/ROC_TPR.dat', np.array(TPR_list), fmt='%.6f')
        np.savetxt(model_save_path+'/ROC_FPR.dat', np.array(FPR_list), fmt='%.6f')
        np.savetxt(model_save_path+'/ROC_Report.dat', np.asarray(report_list), fmt='%.6f')
        

    
    return 0

if __name__ == '__main__':
    sys.exit(main())
