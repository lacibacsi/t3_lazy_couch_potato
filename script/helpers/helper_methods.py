#! /usr/bin/env python

# generic and static helper functions for regularly used tasks

#import rospy
from geometry_msgs.msg import Twist


def CreateTwist(linear_velocity, angular_velocity):
    '''
        Creates a new Twist message based on the input parameter
        Only the linear.x and angular.z values are populated
        Accepts all values

        Input: linear and angular velocity: Float
        Output: new Twist message
    '''
    message = Twist()
    message.linear.x = linear_velocity
    message.angular.z = angular_velocity
    return message
