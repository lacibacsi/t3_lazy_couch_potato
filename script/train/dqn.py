#! usr/bin/env python

import os
import random
import numpy as np
import rospy
import wandb

from state import State
import replay

import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers


#from agent_class import agent_class

class DeepQNetwork:
    '''
        Deep Q learning model for turtlebot autonomous navigation training

        Intuition based on: Bosello et al. (2022) Train in Austria, Race in Montecarlo: Generalized
        RL for Cross-Track F1tenth LIDAR-Based Races
        code: https://github.com/MichaelBosello/f1tenth-RL

        Implementation:
            this class now only uses the 1CNN network with 2 1D convolutional layers, with a flatten and a dense. 
            activation function is relu for the CNN and Dense layers


    '''

    def __init__(self, states, actions):
        '''
            Constructor for the dqn class. Sets private properties, sets up folder for storing model interim and final results 
            This method also calls the internal initialization method that sets the replay buffer, tensorboard directory, etc. 
        '''
        self.path = rospy.get_param('/t3_lazy_couch_potato_v0/dqn_directory')
        self.epsilon = rospy.get_param('/t3_lazy_couch_potato_v0/epsilon')
        self.alpha = rospy.get_param('/t3_lazy_couch_potato_v0/alpha')
        self.gamma = rospy.get_param('/t3_lazy_couch_potato_v0/gamma')

        self.replay_buffer = replay.ReplayMemory(self.path)

        self.actions = []
        for i in range(actions.n):
            self.actions.append(i)
        self.state = states

        self.tensorboard_dir = self.path + "/tensorboard/"
        self.target_model_update_freq = rospy.get_param(
            '/t3_lazy_couch_potato_v0/target_model_update_freq')
        self.checkpoint_dir = self.path + '/models/'
        self.batch_size = rospy.get_param(
            '/t3_lazy_couch_potato_v0/batch_size')
        self.history_length = rospy.get_param(
            '/t3_lazy_couch_potato_v0/history_length')

        self.epsilon_min = rospy.get_param(
            '/t3_lazy_couch_potato_v0/epsilon')

        self.__setup__(len(self.actions),
                       self.state.n, self.tensorboard_dir)

    def __setup__(self, num_actions, state_size, tensorboard_dir, model=None):
        '''
            internal setup method. if not used through the 'agent' injection, this can be the constructor for the class
        '''
        self.num_actions = num_actions
        self.state_size = state_size

        self._states = State()

        # not using camera in this version
        #self.lidar_to_image = args.lidar_to_image
        #self.image_width = args.image_width
        #self.image_height = args.image_height

        # velocity is not used / relevant
        # self.add_velocity = args.add_velocity

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.behavior_net = self.__build_q_net()
        self.target_net = self.__build_q_net()

        model_as_string = []
        self.target_net.summary(print_fn=lambda x: model_as_string.append(x))
        "\n".join(model_as_string)

        summary_writer = tf.summary.create_file_writer(tensorboard_dir)
        with summary_writer.as_default():
            tf.summary.text('model', model_as_string, step=0)

        if model is not None:
            self.target_net.load_weights(model)
            self.behavior_net.set_weights(self.target_net.get_weights())

    def __build_q_net(self):
        return self.__build_cnn1D()

    def __build_cnn1D(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length))
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu',
                          kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu',
                          kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu',
                         kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear',
                                   kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.epsilon),
                      loss=losses.Huber())  # loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def save(self):
        print("saving..")
        self.target_net.save_weights(self.checkpoint_dir)
        self.replay_buffer.save()
        print("saved")

    def inference(self, state):
        state = state.reshape((-1, self.state_size, self.history_length))

        return np.asarray(self.behavior_predict(state)).argmax(axis=1)

    def train(self, batch, step_number):
        '''
            Trains the model 
        '''
        old_states = np.asarray([sample.old_state.get_data()
                                for sample in batch])
        new_states = np.asarray([sample.new_state.get_data()
                                for sample in batch])
        actions = np.asarray([sample.action for sample in batch])
        rewards = np.asarray([sample.reward for sample in batch])
        is_terminal = np.asarray([sample.terminal for sample in batch])

        q_new_state = np.max(self.target_predict(new_states), axis=1)
        target_q = rewards + (self.gamma*q_new_state * (1-is_terminal))
        one_hot_actions = tf.keras.utils.to_categorical(
            actions, self.num_actions)  # using tf.one_hot causes strange errors

        loss = self.gradient_train(old_states, target_q, one_hot_actions)

        if step_number % self.target_model_update_freq == 0:
            self.behavior_net.set_weights(self.target_net.get_weights())

        return loss

    @tf.function
    def target_predict(self, state):
        return self.target_net(state)

    @tf.function
    def behavior_predict(self, state):
        return self.behavior_net(state)

    @tf.function
    def gradient_train(self, old_states, target_q, one_hot_actions):
        with tf.GradientTape() as tape:
            q_values = self.target_net(old_states)
            current_q = tf.reduce_sum(tf.multiply(
                q_values, one_hot_actions), axis=1)
            loss = losses.Huber()(target_q, current_q)

        variables = self.target_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.target_net.optimizer.apply_gradients(zip(gradients, variables))

        return loss

     # agent class virtual methods
    def act(self, state, epsilon):
        '''
            Based on the current state and the agent's settings, returns the selected action             
        '''
        return self.chooseAction(state, epsilon)

    def chooseAction(self, state, epsilon):
        '''
            Returns a selected action - either random or the one with the maximum known reward
        '''
        rospy.loginfo(state)
        if random.random() < epsilon:
            action = random.randrange(len(self.actions))
        else:
            action = self.inference(state.getdata())
        return action

    def learn(self, step_number):
        batch = self.replay_buffer.draw_batch(self.batch_size)
        loss = self.train(batch, step_number)
        return loss

    def add_sample(self, old_state, action, reward, state, done):
        self.replay_buffer.add_sample(replay.Sample(
            old_state, action, reward, state, done))
