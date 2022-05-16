#! /usr/bin/env python

import rospy
from t3_lazy import t3_lazy
from train.qlearn import QLearn
from train.dqn import DeepQNetwork

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
    # commenting out as it's not really working for multiple training algorithms
    # module = __import__(agent_name)
    # agent_class = getattr(module, agent_class_name)

    # creating the wrapper class
    # TODO: figure out why this does not work in a generic way
    if agent_name == 'qlearn':
        rospy.logwarn('training is set to qlearn. ')
        robot = t3_lazy(QLearn)
    elif agent_name == 'dqn':
        rospy.logwarn('training is set to dqn. ')
        robot = t3_lazy(DeepQNetwork)
    else:
        raise ValueError('invalid train config')
    # robot = t3_lazy(agent_class)

    rospy.logwarn('Gym environment done. ')
    rospy.logwarn('Agent is ' + agent_name)

    if agent_name == 'qlearn':
        robot.run()
    else:
        robot.runDQN()
