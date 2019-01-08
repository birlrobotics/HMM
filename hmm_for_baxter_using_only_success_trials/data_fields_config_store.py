data_fields_store = {
    "endpoint_pose": [
        '.endpoint_state.pose.position.x',
        '.endpoint_state.pose.position.y',
        '.endpoint_state.pose.position.z',
        '.endpoint_state.pose.orientation.x',
        '.endpoint_state.pose.orientation.y',
        '.endpoint_state.pose.orientation.z',
        '.endpoint_state.pose.orientation.w'
    ],

    "endpoint_twist": [
        '.endpoint_state.twist.linear.x',
        '.endpoint_state.twist.linear.y',
        '.endpoint_state.twist.linear.z',
        '.endpoint_state.twist.angular.x',
        '.endpoint_state.twist.angular.y',
        '.endpoint_state.twist.angular.z',
    ],
 
   'wrench': [
         '.wrench_stamped.wrench.force.x',
         '.wrench_stamped.wrench.force.y',
         '.wrench_stamped.wrench.force.z',
         '.wrench_stamped.wrench.torque.x',
         '.wrench_stamped.wrench.torque.y',
         '.wrench_stamped.wrench.torque.z',
    ] 
}
