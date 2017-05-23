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
                 linewidth='3.0',
                 fontsize= 50,
                 xaxis_interval=0.005,
                 xlabel= 'time',
                 ylabel = 'log likelihood'):
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
    plt.ylim(-200000,200000)
    
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])*xaxis_interval).tolist()
        if label_string[i-1] == 'threshold':
            plt.plot(O, data, label=label_string[i-1],linewidth=3, linestyle = '--', mfc ="grey")
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

    success_trail_num_train = 23
    success_trail_num_test = 1

    threshold_constant = 3

    threshold_offset = 100

    n_state = joblib.load(model_save_path+'/model_decision/n_state.pkl')

    CV_index = [1,2]


    confusion_matrix = [[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]

    detection_time = [[],[],[],[]]
    

    
   
    
    #########--- load the success Data Index And Label String----###################
    path_index_name = []

    for i in range(21,45):
        if i+1 <= 9:
            post_str = '0'+str(i+1)
        else:
            post_str = str(i+1)
        path_index_name.append('20121127-HIROSA-S-'+post_str)


    for o in range(24):
        Success_Data_Train = []
        Success_Index_Train = []
        Success_Label_String_Train = []

        Success_Data_Test = []
        Success_Index_Test = []
        Success_Label_String_Test = []



        
        CV_Fold_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        CV_Fold_2 = [10,11,12,13,14,15,16,17,18,19]

        if o == 0:
            CV_Train = CV_Fold_1
            CV_Test = CV_Fold_2
        else:
            CV_Train = CV_Fold_2
            CV_Test = CV_Fold_1




        ##########-----loading the Sucess trails data-----------############################
        ## Success_Data_Train[trails][subtask][time_index,feature]
        ## Success_Index_Train[trails][subtask]
        ## Success_Label_String[trails]
        print "Choosing CV index"
        for i in range(24):
            if not i==o:

                data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i],
                                                    preprocessing_scaling=preprocessing_scaling,
                                                    preprocessing_normalize=preprocessing_normalize,
                                                    norm=norm_style)
                Success_Data_Train.append(data_tempt)
                Success_Index_Train.append(index_tempt)
                Success_Label_String_Train.append("Success Trail"+ path_index_name[i])
                print"%d " %i

            else:
                data_tempt, index_tempt = load_data(path=success_path+"/"+path_index_name[i])
                Success_Data_Test.append(data_tempt)
                Success_Index_Test.append(index_tempt)
                Success_Label_String_Test.append("Success Trail"+ path_index_name[i])          





        ######################################################################################


        ######-------loading the HMM Models to list[] ------#################################

        ## Subtasks i hmm model list
        ## model_i[trails]  
        model_1_list = []
        model_2_list = []
        model_3_list = []
        model_4_list = []


        # loading the models
        for i in range(success_trail_num_train):
            if not os.path.isdir(success_path+'/'+path_index_name[i]):
                print success_path+'/'+path_index_name[i]+" do not exit"
                print "Please train the hmm model first, and check your model folder"

            model_1_list.append(joblib.load(model_save_path+'/'+path_index_name[i]+"/model_s1.pkl"))
            model_2_list.append(joblib.load(model_save_path+'/'+path_index_name[i]+"/model_s2.pkl"))
            model_3_list.append(joblib.load(model_save_path+'/'+path_index_name[i]+"/model_s3.pkl"))
            model_4_list.append(joblib.load(model_save_path+'/'+path_index_name[i]+"/model_s4.pkl"))


        ###############################################################################


        ###---------------get the minimun index in every subtask from trails-------#########
        Success_min_index_train = []
        index_tempt = []
        for j in range(4):
            index_tempt = []
            for i in range(success_trail_num_train):
                index_tempt.append(Success_Index_Train[i][j])
            Success_min_index_train.append(min(index_tempt))


        ######-----------training and getting the threshold ----################################
        # Training with left n-1 out
        model_1_end_mean = []
        model_2_end_mean = []
        model_3_end_mean = []
        model_4_end_mean = []
        for n in range(success_trail_num_train):
            model_1_log = []
            model_2_log = []
            model_3_log = []
            model_4_log = []
            for i in range(success_trail_num_train):
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
        model_1_log_full = []
        model_2_log_full = []
        model_3_log_full = []
        model_4_log_full = []

        model_log_full = []
        # Training with left one out
        if not os.path.isdir(model_save_path+"/figure/state_classifcation"):
            os.makedirs(model_save_path+"/figure/state_classifcation")

        for i in range(success_trail_num_test):
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
                    data_1 = best_model[0].score(Success_Data_Test[i][j][:k+1,:])
                    data_2 = best_model[1].score(Success_Data_Test[i][j][:k+1,:])
                    data_3 = best_model[2].score(Success_Data_Test[i][j][:k+1,:])
                    data_4 = best_model[3].score(Success_Data_Test[i][j][:k+1,:])


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
            print "computing the classification log curves of testing (%d/%d)"%(i+1,success_trail_num_test)
            print "Detection Time: %d %d %d %d" %(detection_time[0][-1],detection_time[1][-1],detection_time[2][-1],detection_time[3][-1])


            # if Classification_Full_Flag:
            #     matplot_list([model_1_log,model_2_log,model_3_log,model_4_log],
            #                  figure_index=i,
            #                  label_string=['Approach','Rotation','Insertion','Mating'],
            #                  title='State Classification'+Success_Label_String_Test[i],
            #                  save=True,
            #                  save_path = model_save_path+"/figure/state_classifcation")
            # else:
            #     matplot_list([model_1_log,model_2_log,model_3_log,model_4_log],
            #                  figure_index=i,
            #                  label_string=['Approach','Rotation','Insertion','Mating'],
            #                  title='Wrong State Classification'+Success_Label_String_Test[i],
            #                  save=True,
            #                  save_path = model_save_path+"/figure/state_classifcation")


            np.savetxt(model_save_path+'/figure/state_classifcation/confusion_matrix.dat', np.asarray(confusion_matrix), fmt='%d')
            np.savetxt(model_save_path+'/figure/state_classifcation/detection_time.dat', np.asarray(detection_time), fmt='%d')


            #plt.show()



        #plt.show()



    

    return 0

if __name__ == '__main__':
    sys.exit(main())
