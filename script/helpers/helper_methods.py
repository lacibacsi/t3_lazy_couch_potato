#! /usr/bin/env python

# generic and static helper functions for regularly used tasks

#import rospy
from math import sqrt
import numpy as np
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


def CheckGoalNear(x, y, x_goal, y_goal):
    '''
        Checks if the input coordinate pairs are within 0.3 m distance

        Input: 2 coordinates with x and y values
        Output: True if the distance between the 2 points is less than 0.3
    '''
    ro = sqrt(pow((x_goal - x), 2) + pow((y_goal - y), 2))
    if ro < 0.3:
        return True

    return False


def MeanOfArraysTwoEnd(nparray, slice_size):
    '''
        Returns the mean of the array's beginning and end. The size of the two slices is 2x slice_size
        i.e. if the slice_size is 3, the mean of the numpy array's first 3 and last 3 items will be returned as a single number

        Input:
            nparray:    input one-dimensional numpy array
            slice_size: size of slice on one end

        Output: the mean of the beginning and end of the array - no check is applied on the size of the slice vs the size of the array        
    '''
    np_size = nparray.size()
    concated = np.concatenate(np.sum(nparray[0:slice_size]), np.sum(
        nparray[np_size - slice_size:np_size]))

    # Note: no check on dimension -> this method only works for one-dimensional array
    return np.mean(concated)[0]
