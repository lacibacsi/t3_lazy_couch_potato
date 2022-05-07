#! /usr/bin/env python

import rospy
from t3_lazy import t3_lazy
from train.qlearn import QLearn

if __name__ == "__main__":

    '''
        Training node:
        uses the t3 lazy class (as all trainings) and trains it with the agent set in training_setup.yaml
        Parameters for the training are stored in the agents respective yaml file

        Usage: 
            1. set agent in training_setup.yaml
            2. create an instance of training.py
            3. hope and debug
    '''

    rospy.init_node('t3_lazy_qlearn', anonymous=True, log_level=rospy.INFO)

    agent_name = rospy.get_param('/t3_lazy_couch_potato_v0/agent_name')

    # we could use the same class, maybe later
    agent_class_name = rospy.get_param('/t3_lazy_couch_potato_v0/agent_class')

    # as close to reflection as it gets
    module = __import__(agent_name)
    agent_class = getattr(module, agent_class_name)

    # print(type(agent_class))

    # creating the wrapper class
    # robot = t3_lazy(agent_class)
    robot = t3_lazy(QLearn)

    rospy.logwarn('Gym environment done. ')
    rospy.logwarn('Agent is ' + agent_name)

    robot.run()
