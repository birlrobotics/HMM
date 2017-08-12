import multiprocessing
import rospy
import copy
from birl_sim_examples.msg import (
    Tag_MultiModal,
)

class TagMultimodalTopicHandler(multiprocessing.Process):
    def __init__(self, interested_data_fields, com_queue):
        multiprocessing.Process.__init__(self)     

        interested_data_fields = copy.deepcopy(interested_data_fields)
    
        # we don't need tag when using HMM.score
        tag_idx = interested_data_fields.index('.tag')
        del(interested_data_fields[tag_idx])
        self.interested_data_fields = interested_data_fields
        self.com_queue = com_queue

    def callback_multimodal(self, data):
        data_header = data.endpoint_state.header
        smach_state = data.tag

        one_frame_data = []
        for field in self.interested_data_fields:
            one_frame_data.append(eval('data'+field))

        self.com_queue.put((one_frame_data, smach_state, data_header))       

    def run(self):
        # set up Subscribers
        rospy.init_node("TagMultimodalTopicHandler", anonymous=True)
        rospy.Subscriber("/tag_multimodal", Tag_MultiModal, self.callback_multimodal)
        rospy.loginfo('/tag_multimodal subscribed')

        while not rospy.is_shutdown():
            rospy.spin()
