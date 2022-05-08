#! /usr/bin/env python

import rospy
import gym
import numpy as np
# from environments import t3_lazy_task_env
from environments.t3_lazy_task_env import T3LazyTaskEnv


# TODO: agent param?
EPISODE_LENGTH = 20
EPISODE_COUNT = 10
SAVE_FREQUENCY = 3


class t3_lazy:
    '''
        Main turtlebot 3 class to train the robot
        Uses a primarily memory-based approach
    '''

    def __init__(self, agent_class_type):
        '''
            Constructor for the robot traning. Sets up private variables and creates the environment

            Input: agent_class that will be used for training
        '''

        self.env = gym.make('TurtleBot3LazyCouch-v0')
        rospy.loginfo('Gym environment created')

        self.state_size = self.env.observation_space.shape
        rospy.loginfo('observartion space: {}'.format(
            str(self.env.observation_space)))

        # self.action_size = self.env.action_space.shape
        #self.action_size = self.env.action_space.n
        self.actions = self.env.action_space

        # creating the agent with the action and state sizes
        self.agent = agent_class_type(self.state_size, self.actions)

        self.highest_reward = -np.inf

    def run(self):
        '''
            Based on the input parameters and agent, runs the training for the Turtlebot3
        '''
        steps = 0
        episodes = 0
        last_state = None

        # load previous training if there is some saved file
        self.agent.load()

        try:
            do_training = True

            rospy.loginfo('training started...')

            while do_training and episodes < EPISODE_COUNT:

                rospy.loginfo('episode no. {}'.format(episodes))

                # for every x episodes, save the results
                # the frequency of saving it is defined by SAVE_FREQUENCY
                if episodes != 0 and episodes % SAVE_FREQUENCY == 0:
                    self.agent.save()
                    rospy.logwarn('agent state saved')

                # set up random position to start with
                self.env.initial_position = {'p_x': np.random.uniform(
                    1, 4), 'p_y': 3.7, 'p_z': 0.05, 'o_x': 0, 'o_y': 0.0, 'o_z': np.random.uniform(0.4, 1), 'o_w': 0.360}
                state = self.env.reset()

                rospy.loginfo('environment reset')

                done = False
                cumulated_reward = 0

                episode_steps = 0
                while not done and episode_steps < EPISODE_LENGTH:
                    steps += 1
                    episode_steps += 1

                    # this is where q-learn, sarsa, dqn etc. will act on the input states...
                    rospy.loginfo('step no. {}'.format(steps))

                    action = self.agent.act(state)

                    # performing the action in the environment
                    # done already contains the check if there has been a crash
                    next_state, reward, done, _ = self.env.step(action)

                    cumulated_reward += reward

                    self.agent.learn(last_state, action, reward, next_state)

                # end of episode
                if done:
                    rospy.logwarn('crashed...')

                # update reward, log info
                episodes += 1

                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                rospy.loginfo("total episode_steps {}, reward {}/{}".format(
                    episode_steps, cumulated_reward, self.highest_reward))
                rospy.loginfo("exploration_rate {}".format(
                    self.agent.exploration_rate))

        finally:
            self.env.close()
            rospy.loginfo("Total episodes: {}".format(episodes))
            rospy.loginfo("Total steps: {}".format(steps))
