#! /usr/bin/env python

import numpy as np
import rospy
from gym.envs.registration import register
from gym import spaces

# our custom robot environment - the task environment inherits from this class
from environments import t3_lazy_robot_env
from helpers import helper_methods

# registering the environment
print(register(
    id='TurtleBot3LazyCouch-v0',
    entry_point='environments.t3_lazy_task_env:T3LazyTaskEnv'
    # timestep_limit=timestep_limit_per_episode,
))


# constants for linear and angular speeds for each action
# based on https://github.com/lukovicaleksa/autonomous-driving-turtlebot-with-reinforcement-learning
# and its accompanying thesis

# TODO: move these to a params file
FORWARD_REWARD = 0.2
RIGHT_REWARD = -0.1
LEFT_REWARD = -0.1

OBSTACLE_CLOSER_REWARD = -0.2
OBSTACLE_FURTHER_REWARD = 0.2

SWITCH_LEFT_RIGHT_REWARD = -0.8

LINEAR_SPEED_FWD = 0.08
ANGULAR_SPEED_FWD = 0.0
LINEAR_SPEED_TURN = 0.06
ANGULAR_SPEED_TURN = 0.4

STATE_STATE_MAX_INDEX = 81 - 1  # TODO: move to param
STATE_STATE_MIN_INDEX = 1 - 1


class T3LazyTaskEnv(t3_lazy_robot_env.T3LazyRobotEnv):
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
        self.last_avg = 0.0

        super(T3LazyTaskEnv, self).__init__()

        # this task environment uses the following observation space
        # based on Lukovic Aleksa's MSc theses
        # there are 4 variables, 2 for obstacle distance and 2 for obstacle position
        # x1-x2 (for distance) values set based on how far the obstacle is:
        #   0: if the obstacle is between 20 cm and 40 cm (<20 is crash)
        #   1: if between 40 and 70 cm
        #   2: if between 70 and 100 cm
        #   values higher than that are not important and hence we don't care when training
        #
        # x3-x4 (for positioning) is set in the following way (simplified approach compared to the source)
        #   0: if the obstacle is closer to the forward axis - left1 and right1 (0 and 25 degrees)
        #   1: if the obstacle is between 25 and 50 degrees
        #   2: if the obstacle is between 50 and 75 degrees
        #
        # hence the observation space size is 3x3x3x3 = 81
        # as all 4 variables are discrete with max 3 values, a tuple of 4 discrete space can be set up
        self.observation_space = spaces.Tuple({'x1': spaces.Discrete(3), 'x2': spaces.Discrete(
            3), 'x3': spaces.Discrete(3), 'x4': spaces.Discrete(3)})

        # self.observation_space = spaces.Box(
        #    np.array([0, 0, 0, 0]), np.array([2, 2, 2, 2]), dtype=np.float32)

        # super(T3LazyTaskEnv, self).__init__()

    def _get_distances(self):
        '''
            Returns an average distance of LIDAR reads for 'sectors' - inspired by Lukovic Aleksa's MSc Thesis
            The full laser scan range read by the robot is split into 6 sectors
            to avoid obstacles we don't really care about the 90 - 75 degree ranges,
            so the forward sector is not uniformly split into the following ranges from left to right:
            - 75 -> - 50 degrees (left)
            - 50 -> -25 degrees
            - 25 -> 0 degrees (forward left)
            0 -> 25 degrees (forward right)
            25 -> 50 degrees
            50 -> 75 degrees (right)

            For each range a mean value is returned as well as the average of all relevant values (-75 to +75 degrees)

            Input: none
            Return: 6 mean distance values from left to right
        '''
        ranges = self.get_laser_scan()

        left3 = np.mean(ranges[50:75])
        left2 = np.mean(ranges[25:50])
        left1 = np.mean(ranges[0:25])
        rigt3 = np.mean(ranges[285:310])
        right2 = np.mean(ranges[310:335])
        right1 = np.mean(ranges[335:360])

        avg = helper_methods.MeanOfArraysTwoEnd(ranges, 75)

        return left3, left2, left1, right1, right2, rigt3, avg

    # overwriting robot obs
    def _get_obs(self):
        '''
            returns the observations by the robot
            which is build up by the following method
            there are 4 variables, 2 for obstacle distance and 2 for obstacle position
            x1-x2 (for distance) values set based on how far the obstacle is for left and right side respectively:
            0: if the obstacle is between 20 cm and 40 cm (<20 is crash)
            1: if between 40 and 70 cm
            2: if between 70 and 100 cm
            values higher than that are not important and hence we don't care when training. 
            Values larger than 100 cm are filtered out

            x3-x4 (for positioning) is set in the following way (simplified approach compared to the source)
            0: if the obstacle is closer to the forward axis - left1 and right1 (0 and 25 degrees)
            1: if the obstacle is between 25 and 50 degrees
            2: if the obstacle is between 50 and 75 degrees

            the method additionally returns the average measured distance from the next obstacle for the total width of the lidar scan 
        '''
        left3, left2, left1, right1, right2, right3, avg = self._get_distances()

        # calculating values for observability space
        x1 = 2
        left_min = min([left1, left2, left3])
        #rospy.logwarn('lefts: {}'.format([left3, left2, left1]))
        #rospy.logwarn('leftmin: {}'.format(left_min))

        if (left_min < 0.70):
            x1 = 1
            if (left_min < 0.40):
                x1 = 0

        x2 = 2
        right_min = min([right1, right2, right3])
        if (right_min < 0.70):
            x2 = 1
            if (right_min < 0.40):
                x2 = 0

        x3 = 2
        if (left_min == left2):
            x3 = 1
        elif (left_min == left1):
            x3 = 0

        x4 = 2
        if (right_min == right2):
            x4 = 1
        elif (right_min == right1):
            x4 = 0

        return (x1, x2, x3, x4, avg)

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

        # 1. if there is a collision, huge negative reward
        # 2. otherwise prefer forward movement
        # 3. add reward / penalty based on x1, x2, x3, x4
        # 4. add reward on whether we're moving away from ostacle

        if not done:
            # 2 - prefer forward motion
            if self.last_action == 0:
                reward = 0.2
            else:
                reward = -0.1

            # 3. add reward / penalty based on x1, x2, x3, x4
            # if x1 or x2 is small there is an obstacle close by
            # if x3 or x4 is small the obstacle is close to the left / right
            # each case is penalized by -0.3 / -0.6
            (x1, x2, x3, x4, avg) = self._get_obs()
            x_penalty_factor = 0.15

            reward = -x_penalty_factor * (2-x1)
            reward = -x_penalty_factor * (2-x2)
            reward = -x_penalty_factor * (2-x3)
            reward = -x_penalty_factor * (2-x4)

            # 4. add reward on whether we're moving away from ostacle
            if self.last_avg > avg:
                rw2 = OBSTACLE_CLOSER_REWARD
            else:
                rw2 = OBSTACLE_FURTHER_REWARD

            reward += rw2

            self.last_avg = avg

            #obs_distance = self._closest_obstacle()
            # if obs_distance < 0.3:
            #    reward -= 0.3
            # elif obs_distance < 0.6:
            #    reward -= 0.1
        else:
            reward = -100

        self.cumulated_reward += reward
        self.cumulated_steps += 1

        return reward

    def _set_action(self, action):
        '''
            Applies the given action to the robot.
            The Turtlebot 3's action space is set up 3 values:
            move forward (0), turn left (1) and turn right (2)

            Input:  action, int
            Output: none
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
