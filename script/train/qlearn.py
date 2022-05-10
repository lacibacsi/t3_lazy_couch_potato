#! usr/bin/env python

import pickle
import random
import os
import rospy
#from agent_class import agent_class


'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''


# class QLearn(agent_class):
class QLearn():
    '''
        qlearn implementation -> used in assessment 2 for the Robotics cours
        Based on the construct code and https://github.com/vmayoral/basic_reinforcement_learning
        some parts and ideas taken from https://github.com/karray/neuroracer 
    '''

    def __init__(self, states, actions):
        '''
            Constructor for the Qlearn class. Sets private properties, sets up folder for storing model interim and final results 
        '''
        self.q = {}
        #self.state_size = state_size

        # TODO: params
        self.path = rospy.get_param('/t3_lazy_couch_potato_v0/qfile_directory')
        self.qvalues_filename = 'qlearn_results'
        self.qfile = os.path.join(self.path, self.qvalues_filename)

        # reading config values from config/qlearn_params.yaml
        epsilon = rospy.get_param('/t3_lazy_couch_potato_v0/epsilon')
        alpha = rospy.get_param('/t3_lazy_couch_potato_v0/epsilon')
        gamma = rospy.get_param('/t3_lazy_couch_potato_v0/epsilon')

        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor

        #self.actions = actions
        # the Discrete action space is int 0 -> n-1, so adding the integers to an iterable to make is easier.
        # Note: this only works for a one-dimensional action space
        self.actions = []
        for i in range(actions.n):
            self.actions.append(i)

        rospy.logwarn('actions converted to {}'.format(self.actions))

        self.state = states

        rospy.logwarn('self state: {}'.format(self.state))

        #self.path = path
        #self.q_file_name = self.path + self.qvalues_filename

        # TODO: replace this with proper ctor chaining
        # super(QLearn, self).__init__(state_size, action_size)

        # required by parent
        self.exploration_rate = epsilon

    def getQ(self, state, action):
        return self.q.get(((state), action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get(((state), action), None)
        if oldv is None:
            self.q[((state), action)] = reward
        else:
            self.q[((state), action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 *
                 mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        return action

    def _learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    # agent class virtual methods
    def act(self, state):
        '''
            Based on the current state and the agent's settings, returns the selected action             
        '''
        return self.chooseAction(state)

    def learn(self, last_state, action, reward, next_state):
        '''
            Implements the main learning algorithm

            Inputs:
                    last_state  - last state space index
                    action      - action selected last
                    reward      - reward received based on last state and performed action
                    next state  - new state space index
        '''
        #rospy.loginfo('learn called, last state: {}'.format(last_state))
        #rospy.loginfo('learn called, next state: {}'.format(next_state))
        #rospy.loginfo('learn called, action: {}'.format(action))
        #rospy.loginfo('learn called, next state: {}'.format(reward))

        return self._learn(last_state, action, reward, next_state)

    def save(self):
        '''
            Saves the interim qvalues. The file gets rewritten every time, this agent does not keep history.
            History is better persisted and analyzed at the caller, training class level
        '''
        #rospy.logwarn('type of q table: {}'.format(type(self.q)))
        with open(self.qfile, mode='wb') as f:
            #rospy.loginfo('q values are: {}'.format(self.q))
            pickle.dump(self.q, f, protocol=pickle.HIGHEST_PROTOCOL)
            #rospy.loginfo('q saved as {}'.format(str(self.q)))

    def load(self):
        '''
            If present, leads the interim qvalues and state information 
        '''
        if os.path.isfile(self.qfile) and os.path.getsize(self.qfile) > 0:
            with open(self.qfile, mode='rb') as f:
                #self.q = f.read()
                self.q = pickle.load(f)
                rospy.loginfo('q loaded as {}'.format(self.q))
        else:
            # file does not exist or is empty
            self.q = {}
