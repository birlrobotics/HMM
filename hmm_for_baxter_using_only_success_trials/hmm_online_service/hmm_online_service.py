#!/usr/bin/env python

import rospy
def run(interested_data_fields, model_save_path, state_amount, deri_threshold):
    from multiprocessing import Queue
    import data_stream_handler_process
    com_queue_of_receiver = Queue()
    process_receiver = data_stream_handler_process.TagMultimodalTopicHandler(
        interested_data_fields,
        com_queue_of_receiver,
    )

    import anomaly_detection_process 
    com_queue_of_anomaly_detection = Queue()
    process_anomaly_detection = anomaly_detection_process.AnomalyDetector(
        model_save_path,
        state_amount,
        deri_threshold,
        com_queue_of_anomaly_detection,    
    )



    process_receiver.start()
    process_anomaly_detection.start()

    while not rospy.is_shutdown():
        try:
            latest_data_tuple = com_queue_of_receiver.get(1)
        except Queue.Empty:
            continue
        latest_data_tuple = com_queue_of_receiver.get()
        com_queue_of_anomaly_detection.put(latest_data_tuple)


    process_receiver.shutdown()
    process_anomaly_detection.shutdown()
    '''
    thread1 = ROSThread(interested_data_fields)  
    thread2 = HMMThreadForAnomalyDetection(model_save_path, state_amount, deri_threshold)
    #thread3 = HMMThreadForStateClassification(model_save_path, state_amount)
    thread1.setDaemon(True)
    thread2.setDaemon(True)
    #thread3.setDaemon(True)
    thread1.start()  
    thread2.start()
    #thread3.start()
    '''
