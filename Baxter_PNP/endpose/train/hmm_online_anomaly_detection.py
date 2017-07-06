#!/usr/bin/env python

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import threading
from sklearn.externals import joblib
import rospy
import copy
from std_msgs.msg import (
    Header
)
from birl_sim_examples.msg import (
    Tag_MultiModal,
    Hmm_Log
)

data_arr = []
hmm_previous_state =0
hmm_state = 0
block_detection = False
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
        global block_detection

        hmm_state = data.tag
        if not hmm_state==hmm_previous_state:
            block_detection = True
            data_arr = []
            rospy.loginfo("state %s->%s"%(hmm_previous_state,hmm_state))
            rospy.loginfo("block until var reaches the boundary")

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

class HMMThread(threading.Thread):
    def __init__(self, model_save_path, state_amount, var_boundary):
        threading.Thread.__init__(self) 

        self.var_boundary = var_boundary

        list_of_expected_log = joblib.load(model_save_path+'/multisequence_model/expected_log.pkl')
        list_of_threshold = joblib.load(model_save_path+'/multisequence_model/threshold.pkl')
        list_of_var_of_log = joblib.load(model_save_path+"/multisequence_model/var_of_log.pkl")

        model_group_by_state = {}
        expected_log_group_by_state = {}
        threshold_group_by_state = {} 
        var_log_group_by_state = {}

        for state_no in range(1, state_amount+1):
            model_group_by_state[state_no] = joblib.load(model_save_path+"/multisequence_model/model_s%s.pkl"%(state_no,))


            # the counterpart simply pushes these data into a list, so for state 1, its data is located in 0.
            expected_log_group_by_state[state_no] = list_of_expected_log[state_no-1]
            threshold_group_by_state[state_no] = list_of_threshold[state_no-1]
            var_log_group_by_state[state_no] = list_of_var_of_log[state_no-1]


        self.model_group_by_state = model_group_by_state

        self.expected_log_group_by_state = expected_log_group_by_state
        self.var_log_group_by_state = var_log_group_by_state 
        self.threshold_group_by_state = threshold_group_by_state 

    def run(self):
        global data_arr
        global hmm_state
        global header
        global block_detection

        hmm_log = Hmm_Log()
        publishing_rate = 50
        r = rospy.Rate(publishing_rate)
        pub = rospy.Publisher("/hmm_online_result", Hmm_Log, queue_size=10)
        rospy.loginfo('/hmm_online_result published')
    
        while not rospy.is_shutdown():
            if hmm_state == 0:
                r.sleep()
                continue

            # no data arrived yet
            data_index = len(data_arr)
            if data_index == 0:
                r.sleep()
                continue

            if block_detection:
                if self.var_log_group_by_state[hmm_state][data_index] > self.var_boundary:
                    rospy.loginfo('var boundary reached, unblock')
                    block_detection = False
                else:
                    r.sleep()
                    continue

            try:    
                hmm_log.expected_log.data = self.expected_log_group_by_state[hmm_state][data_index-1]
                hmm_log.threshold.data = self.threshold_group_by_state[hmm_state][data_index-1]
                hmm_log.current_log.data = self.model_group_by_state[hmm_state].score(data_arr)
                if hmm_log.current_log.data > hmm_log.threshold.data:
                    hmm_log.event_flag = 1
                else:
                    hmm_log.event_flag = 0

                hmm_log.header = header
                pub.publish(hmm_log)
            except IndexError:
                rospy.loginfo('received data is longer than the threshold. DTW needed.')

            r.sleep()

        return 0

    
def run(interested_data_fields, model_save_path, state_amount, var_boundary):
    rospy.init_node("hmm_online_parser", anonymous=True)
    thread1 = ROSThread(interested_data_fields)  
    thread2 = HMMThread(model_save_path, state_amount, var_boundary)
    thread1.setDaemon(True)
    thread2.setDaemon(True)
    thread1.start()  
    thread2.start()
    rospy.spin()
    return 0
    
