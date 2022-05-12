#! /usr/bin/env python

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from environments.t3_lazy_task_env import T3LazyTaskEnv
import rospy
import gym
import os
import numpy as np
import math
import random
# from environments import t3_lazy_task_env

# to be able to provide random orientation

# TODO: agent param?
EPISODE_LENGTH = 100
EPISODE_COUNT = 1000
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

        self.path = rospy.get_param('/t3_lazy_couch_potato_v0/qfile_directory')
        self.reward_filename = 'episode_reward'
        self.episodefile = os.path.join(self.path, self.reward_filename)

        self.env = gym.make('TurtleBot3LazyCouch-v0')
        rospy.loginfo('Gym environment created')

        #self.state_size = self.env.observation_space.shape
        self.states = self.env.observation_space

        rospy.loginfo('observartion space: {}'.format(
            str(self.env.observation_space)))

        # self.action_size = self.env.action_space.shape
        #self.action_size = self.env.action_space.n
        self.actions = self.env.action_space

        # creating the agent with the action and state sizes
        #self.agent = agent_class_type(self.state_size, self.actions)
        self.agent = agent_class_type(self.states, self.actions)

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
                orientation = self.get_random_orientation()
                # self.env.initial_position = {'p_x': 0.0, 'p_y': 0.0, 'p_z': 0.0,
                #                             'o_x': 0, 'o_y': 0.0, 'o_z': np.random.uniform(0.4, 1), 'o_w': 0}

                self.env.initial_position = {'p_x': 0.0, 'p_y': 0.0, 'p_z': 0.0,
                                             'o_x': orientation[0], 'o_y': orientation[1], 'o_z': orientation[2], 'o_w': orientation[3]}
                state = self.env.reset()

                rospy.loginfo('environment reset')

                done = False
                cumulated_reward = 0

                episode_steps = 0
                while not done and episode_steps < EPISODE_LENGTH:
                    steps += 1
                    episode_steps += 1

                    # this is where q-learn, sarsa, dqn etc. will act on the input states...
                    #rospy.loginfo('step no. {}'.format(steps))

                    action = self.agent.act(state)

                    # performing the action in the environment
                    # done already contains the check if there has been a crash
                    next_state, reward, done, _ = self.env.step(action)

                    cumulated_reward += reward

                    self.agent.learn(last_state, action, reward, next_state)
                    last_state = next_state

                # end of episode
                if done:
                    rospy.logwarn('crashed...')
                    # rospy.logwarn('distance values: {}'.format(
                    #    self.env._get_distances()))
                    #rospy.logwarn('x values: {}'.format(self.env._get_obs()))

                self.save_episode_reward(
                    episodes, cumulated_reward, episode_steps)

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

    def save_episode_reward(self, episode_number, reward, steps):
        '''
            appends the episode number and the reward to the end of the training result\episode_reward file
        '''
        with open(self.episodefile, mode='a') as f:
            row = str(episode_number) + ', ' + \
                str(reward) + ', ' + str(steps) + '\n'
            f.write(row)

    def get_random_orientation(self):
        '''
            returns the robot orientation for a random yawn between -90 and 90 degrees
        '''

        yaw_angle = random.randint(90, 270)
        print(yaw_angle)
        yaw_rad = math.radians(yaw_angle)
        orientation = quaternion_from_euler(0, 0, yaw_rad)
        return orientation
