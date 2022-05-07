#! /usr/bin/evn python

class agent_class():
    '''
        Abstract agent class containing only virtual methods 
        These methods are used by the robot training class 
        and are implemented by the individual RL algorithms
    '''

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # properties to implement
        # self.exploration_rate

    def act(self, state):
        '''
            Based on the current state and the agent's settings, returns the selected action             
        '''
        raise NotImplementedError()

    def learn(self, last_state, action, reward, next_state):
        '''
            Implements the main learning algorithm

            Inputs:
                    last_state  - last state space index
                    action      - action selected last
                    reward      - reward received based on last state and performed action
                    next state  - new state space index
        '''
        raise NotImplementedError()
