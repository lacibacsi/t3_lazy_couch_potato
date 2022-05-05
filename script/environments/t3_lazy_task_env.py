#! /usr/bin/env python

import numpy as np
import rospy
from gym.envs.registration import register
from gym import spaces

# our custom robot environment - the task environment inherits from this class
from environments import t3_lazy_robot_env

# registering the environment
print(register(
    id='TurtleBot3LazyCouch-v0',
    entry_point='t3_lazy_task_evk:T3LazyTrainEnv'
    # timestep_limit=timestep_limit_per_episode,
))


# constants for linear and angular speeds for each action
# based on https://github.com/lukovicaleksa/autonomous-driving-turtlebot-with-reinforcement-learning
# and its accompanying thesis

# TODO: move these to a params file
FORWARD_REWARD: 0.2
RIGHT_REWARD: -0.1
LEFT_REWARD: -0.

LINEAR_SPEED_FWD = 0.8
ANGULAR_SPEED_FWD = 0.0
LINEAR_SPEED_TURN = 0.06
ANGULAR_SPEED_TURN = 0.4


class T3LazyTrainEnv(t3_lazy_robot_env.T3LazyRobotEnv):
    '''
        Custom train environment based on:
        OpenAI example: https://bitbucket.org/theconstructcore/openai_ros and 
        NeuroRacer: https://github.com/karray/neuroracer

        In general it uses a simple reward mechanism and a discretized observation space based on the above sources
    '''

    def __init__(self):

        # fixed set of actions fopr the robot:
        # 0: move forward
        # 1: turn left
        # 2: turn right
        self.action_space = spaces.Discrete(3)

        self.rate = rospy.Rate(10)  # setting to 10Hz for now
        self.cumulated_steps = 0.0

    def _get_distances(self):
        '''
            Returns an average distance of LIDAR reads for 'sectors'
            The full laser scan range read by the robot is split into 5 sectors
            to avoid obstacles we don't really care about the 90 - 75 degree ranges,
            so the forward sector is not uniformly split into the following ranges from left to right:
            - 75 -> - 55 degrees (left)
            - 55 -> -30 degrees
            - 30 -> 30 degrees
            30 -> 55 degrees
            55 -> 75 degrees (right)

            For each range a mean value is returned

            Input: none
            Return: 5 mean distance values from left to right
        '''
        ranges = self.get_laser_scan()

        most_left = np.mean(ranges[55:75])
        left = np.mean(ranges[30:55])

        right = np.mean(ranges[305:330])
        most_right = np.mean(ranges[285:305])

        forward = np.concatenate(np.sum(ranges[0:30]), np.sum(ranges[330:359]))
        forward = np.mean(forward)

        return most_left, left, forward, right, most_right

    # virtual methods from robot envrionment
    def _set_init_pose(self):
        '''
            Sets the Robot in its init pose
        '''
        self.move_base(0, 0)
        return True

    def _init_env_variables(self):
        '''
            Inits variables needed to be initialised each time we reset at the start of an episode.
        '''
        self.cumulated_reward = 0.0
        self.last_action = 0    # forward
        self._episode_done = False

    def _compute_reward(self, observations, done):
        '''
            Calculates the reward to give based on the observations given.
        '''

        # extremely simple for now - this needs to be reworked big time
        # TODO: implement proper reward function

        # 1. if there is a collision, huge negative reward
        # 2. otherwise prefer forward movement
        # 3. just a try now -> penalize if the obstacle is very close

        if not done:
            if self.last_action == 0:
                reward = 0.2
            else:
                reward = -0.1
            obs_distance = self._closest_obstacle()
            if obs_distance < 0.3:
                reward -= 0.3
            elif obs_distance < 0.6:
                reward -= 0.1
        else:
            reward = -100

        self.cumulated_reward += reward
        self.cumulated_steps += 1

        return reward

    def _set_action(self, action):
        '''
            Applies the given action to the simulation.
        '''
        if action == 0:
            # forward
            self.move_base(LINEAR_SPEED_FWD, ANGULAR_SPEED_FWD)
        elif action == 1:
            # left turn
            self.move_base(LINEAR_SPEED_FWD, -1 * ANGULAR_SPEED_FWD)
        elif action == 2:
            # right turn
            self.move_base(LINEAR_SPEED_FWD, ANGULAR_SPEED_FWD)
        else:
            raise ValueError('invalid action')

        self.last_action = action
        self.rate.sleep()
