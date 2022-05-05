#! /usr/bin/env python

# generic and static helper functions for regularly used tasks

#import rospy
from math import sqrt
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

# Check - goal near


def CheckGoalNear(x, y, x_goal, y_goal):
    '''
        Checks if the input coordinate pairs are within 0.3 m distance

        Input: 2 coordinates with x and y values
        Output: True if the distance between the 2 points is less than 0.3
    '''
    ro = sqrt(pow((x_goal - x), 2) + pow((y_goal - y), 2))
    if ro < 0.3:
        return True
    else:
        return False
