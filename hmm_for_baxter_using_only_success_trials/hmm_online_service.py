#!/usr/bin/env python

import os
import threading
from sklearn.externals import joblib
import rospy
import copy
from std_msgs.msg import (
    Header,
    Float32
)
from birl_sim_examples.msg import (
    Tag_MultiModal,
    Hmm_Log
)

import copy

data_arr = []
hmm_previous_state =0
hmm_state = 0
prev_diff = None
header = Header()

class ROSThread(threading.Thread):
    def __init__(self, interested_data_fields):
        threading.Thread.__init__(self)     

        interested_data_fields = copy.deepcopy(interested_data_fields)
    
        # we don't need tag when using HMM.score
        tag_idx = interested_data_fields.index('.tag')
        del(interested_data_fields[tag_idx])
        self.interested_data_fields = interested_data_fields

    def callback_multimodal(self, data):
        global hmm_state
        global data_arr
        global header
        global hmm_previous_state
        global prev_diff 

        header = data.endpoint_state.header
        hmm_state = data.tag
        if not hmm_state==hmm_previous_state:
            prev_diff = None
            data_arr = []
            rospy.loginfo("state %s->%s"%(hmm_previous_state,hmm_state))

        one_frame_data = []
        for field in self.interested_data_fields:
            one_frame_data.append(eval('data'+field))
       
        data_arr.append(one_frame_data) 
        hmm_previous_state = hmm_state

    def run(self):
        # set up Subscribers
        rospy.Subscriber("/tag_multimodal", Tag_MultiModal, self.callback_multimodal)
        rospy.loginfo('/tag_multimodal subscribed')

        while not rospy.is_shutdown():
            rospy.spin()

class HMMThreadForAnomalyDetection(threading.Thread):
    def __init__(self, model_save_path, state_amount, deri_threshold):
        threading.Thread.__init__(self) 

        self.deri_threshold = deri_threshold


        list_of_expected_log = joblib.load(model_save_path+'/expected_log.pkl')
        list_of_threshold = joblib.load(model_save_path+'/threshold.pkl')
        list_of_std_of_log = joblib.load(model_save_path+"/std_of_log.pkl")

        model_group_by_state = {}
        expected_log_group_by_state = {}
        threshold_group_by_state = {} 
        std_log_group_by_state = {}

        for state_no in range(1, state_amount+1):
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))


            # the counterpart simply pushes these data into a list, so for state 1, its data is located in 0.
            expected_log_group_by_state[state_no] = list_of_expected_log[state_no-1]
            threshold_group_by_state[state_no] = list_of_threshold[state_no-1]
            std_log_group_by_state[state_no] = list_of_std_of_log[state_no-1]


        self.model_group_by_state = model_group_by_state

        self.expected_log_group_by_state = expected_log_group_by_state
        self.std_log_group_by_state = std_log_group_by_state 
        self.threshold_group_by_state = threshold_group_by_state 

    def get_anomaly_detection_msg(self):
        global data_arr
        global hmm_state
        global header
        global prev_diff 

        data_arr_copy = copy.deepcopy(data_arr)

        data_index = len(data_arr_copy)
        if data_index == 0:
            return None

        hmm_state_copy = hmm_state

        if hmm_state_copy == 0:
            return None

        hmm_log = None

        try:    
            expected_log = self.expected_log_group_by_state[hmm_state_copy][data_index-1]
            threshold = self.threshold_group_by_state[hmm_state_copy][data_index-1]
            current_log = self.model_group_by_state[hmm_state_copy].score(data_arr_copy)

            now_diff = current_log-expected_log
    
            if prev_diff is not None:
                hmm_log = Hmm_Log()

                hmm_log.current_log.data = current_log 
                hmm_log.expected_log.data = expected_log
                hmm_log.threshold.data = threshold
                hmm_log.diff_btw_curlog_n_thresh.data = now_diff
                hmm_log.deri_of_diff_btw_curlog_n_thresh.data = now_diff-prev_diff

                if abs(now_diff-prev_diff) < self.deri_threshold:
                    hmm_log.event_flag = 1
                else:
                    hmm_log.event_flag = 0
                hmm_log.header = header

            prev_diff = now_diff

        except IndexError:
            rospy.loginfo('received data is longer than the threshold. DTW needed.')
            hmm_log = None
            prev_diff = None
            
        return hmm_log

    def run(self):
        publishing_rate = 10 
        r = rospy.Rate(publishing_rate)

        anomaly_topic_pub = rospy.Publisher("/hmm_online_result", Hmm_Log, queue_size=10)
        rospy.loginfo('/hmm_online_result published')

        while not rospy.is_shutdown():


            hmm_log = self.get_anomaly_detection_msg()
            if hmm_log is not None:
                anomaly_topic_pub.publish(hmm_log)

            r.sleep()

        return 0


class HMMThreadForStateClassification(threading.Thread):
    def __init__(self, model_save_path, state_amount):
        threading.Thread.__init__(self) 

        model_group_by_state = {}
        for state_no in range(1, state_amount+1):
            model_group_by_state[state_no] = joblib.load(model_save_path+"/model_s%s.pkl"%(state_no,))
        self.model_group_by_state = model_group_by_state


    def get_state_classification_msgs_group_by_state(self):
        global data_arr
        global hmm_state

        data_arr_copy = copy.deepcopy(data_arr)

        data_index = len(data_arr_copy)
        if data_index == 0:
            return None

        hmm_state_copy = hmm_state

        if hmm_state_copy == 0:
            return None

        msgs_group_by_state = {}
        for state_no in self.model_group_by_state:
            msgs_group_by_state[state_no] = self.model_group_by_state[state_no].score(data_arr_copy)
        return msgs_group_by_state

    def run(self):
        publishing_rate = 10 
        r = rospy.Rate(publishing_rate)

        state_log_curve_pub = {}
        for state_no in self.model_group_by_state:
            topic_name = "/hmm_log_curve_of_state_%s"%(state_no,)
            state_log_curve_pub[state_no] = rospy.Publisher(topic_name, Float32, queue_size=10)
            rospy.loginfo('%s published'%(topic_name,))
        

        while not rospy.is_shutdown():

            msgs_group_by_state = self.get_state_classification_msgs_group_by_state()
            if msgs_group_by_state is not None:
                for state_no in msgs_group_by_state:
                    state_log_curve_pub[state_no].publish(msgs_group_by_state[state_no]) 

            r.sleep()

        return 0
    
def run(interested_data_fields, model_save_path, state_amount, deri_threshold):
    rospy.init_node("hmm_online_service", anonymous=True)
    thread1 = ROSThread(interested_data_fields)  
    thread2 = HMMThreadForAnomalyDetection(model_save_path, state_amount, deri_threshold)
    thread3 = HMMThreadForStateClassification(model_save_path, state_amount)
    thread1.setDaemon(True)
    thread2.setDaemon(True)
    thread3.setDaemon(True)
    thread1.start()  
    thread2.start()
    thread3.start()
    rospy.spin()
    return 0
    
