#! /usr/bin/env python

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from environments.t3_lazy_task_env import T3LazyTaskEnv
import rospy
import gym
import os
import numpy as np
import math
import random
import wandb
# from environments import t3_lazy_task_env

# wandb settings
WANDB_PROJECT = 't3_lazy_couch'
WANDB_ENTITY = 'lacibacsi'


# TODO: agent param?
EPISODE_LENGTH = 200
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

        # setting up the environment to use lidar readings for dqn, biggest hack, this needs rework too
        if str(agent_class_type) == 'DeepQNetwork':
            # this will return the observation space as lidars
            self.env.use_lidar_as_space = True

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

        # initiailizing wandb project
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

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

                    # next state also contains the average obstacle distance
                    # that is not needed for the q table
                    next_state = (
                        next_state[0], next_state[1], next_state[2], next_state[3])

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

    def runDQN(self, eval_with_epsilon=None):
        '''
            Runs the dqn training for Turtlebot3
            This method should not exist, should be part of the generic 'run' method
            TODO: merge the two together

            Input: 
                eval_with_epsilon: if set, no training is done, but prediction with the set epsilon
                set it to 1 to have a full prediction-based only method
        '''

        steps = 0
        episodes = 0
        last_state = None
        train_epsilon = self.agent.epsilon
        epsilon_min = self.agent.epsilon_min

        # sets if the run is in a training mode or not
        do_training = True if eval_with_epsilon is None else False
        epoch_total_score = 0

        # the agent passed to this class is a DeepQNetwork that has been initialized
        # and has a replay memory, batch count, action and observation space etc.

        rospy.loginfo('training started...')

        # while do_training and episodes < EPISODE_COUNT: - don't check for training here as we need this to playback
        while episodes < EPISODE_COUNT:

            done = False
            cumulated_reward = 0
            episode_steps = 0
            episode_losses = []

            # important - the custom environment's state is the observation state of 5 values - x1, x2, x3, x4
            # here, however, we want to get the full lidar reading, hence using the laser scan readings
            try:

                state = self.env.get_laser_scan()

                while not done and episode_steps < EPISODE_LENGTH:
                    steps += 1
                    episode_steps += 1

                    # epsilon selection and update
                    if do_training:
                        epsilon = train_epsilon
                        if train_epsilon > epsilon_min:
                            train_epsilon = train_epsilon * self.agent.epsilon_decay
                            if train_epsilon < epsilon_min:
                                train_epsilon = epsilon_min
                        else:
                            epsilon = eval_with_epsilon

                    # action selection
                    # state is known by the actor, no need to pass it
                    action = self.agent.act(state, epsilon)

                    old_state = state   # storing current readings
                    # for i in range(0, args.history_length * (args.repeat_action + 1)):
                    # not using these params, but using the defaul 2 for now for both replay and history
                    for i in range(0, 2 * (2 + 1)):
                        if episode_steps % SAVE_FREQUENCY == 0:
                            save_net = True

                        next_state, reward, done = self.env.step(action)
                        cumulated_reward += reward

                        # train
                        if do_training and old_state is not None:
                            if episode_steps > self.agent.observation_steps:
                                loss = self.agent.learn()
                                episode_losses.append(loss)
                                # log loss here
                                wandb.log(
                                    {"episode": episodes, "episode steps": episode_steps, "loss": loss})

                        if done:
                            break

                    # Record experience in replay memory
                    if do_training and old_state is not None:
                        self.agent.add_sample(
                            old_state, action, reward, state, done)

                    if done:
                        state = None

                # end of episode
                if done:
                    rospy.logwarn('crashed...')

                if save_net:
                    self.agent.save()

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
            logs the reward to wandb project as well
        '''
        with open(self.episodefile, mode='a') as f:
            row = str(episode_number) + ', ' + \
                str(reward) + ', ' + str(steps) + '\n'
            f.write(row)

        wandb.log({"reward": reward, "episode steps": steps})

    def get_random_orientation(self):
        '''
            returns the robot orientation for a random yawn between -90 and 90 degrees
        '''

        yaw_angle = random.randint(90, 270)
        print(yaw_angle)
        yaw_rad = math.radians(yaw_angle)
        orientation = quaternion_from_euler(0, 0, yaw_rad)
        return orientation
