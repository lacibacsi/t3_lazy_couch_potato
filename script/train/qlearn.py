#! usr/bin/env python

import random
from agent_class import agent_class


'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''


class QLearn(agent_class):
    '''
        qlearn implementation -> used in assessment 2 for the Robotics cours
        Based on the construct code and https://github.com/vmayoral/basic_reinforcement_learning
        some parts and ideas taken from https://github.com/karray/neuroracer 
    '''

    def __init__(self, state_size, action_size, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        #self.actions = actions
        self.actions = action_size
        self.state = state_size

        # TODO: replace this with proper ctor chaining
        super.__init__(self, state_size, action_size)

        # required by parent
        self.exploration_rate = epsilon

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

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
        return self._learn(last_state, action, reward, next_state)
