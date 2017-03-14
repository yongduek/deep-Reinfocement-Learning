'''
   One-step Deep SARSA-learning
   SARSA network: input=(state/observation, action), output = Q(state, action)
   for each episode
        initialize
        observe state
        select action using policy based on Q
        for t=1, ..., T
            do_action, observe reward and next state
            selection action2 using policy

            Q[s,a] <- Q[s,a] + alpha (r + gamma * Q[s',a'] - Q[s,a])

            # do this as update
            y = reward             for terminal state
                reward + gamma Q(s', a')
            perform (y - Q(theta))^2 minimization; this updates Q online. so this is on-policy algorithm
            #
        endfor
    endfor
'''

import gym
from gym.envs.registration import register
import numpy as np
from collections import deque

import tensorflow as tf
import tflearn

register(
    id='FrozenLake-v44',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)
#reward_threshold=0.78, # optimum = .8196


def onehot(n, x):
    '''
    :param x:
    :return: a n row vector whose x-th elem. is 1
    '''
    return np.identity(n)[x]
#


class tfnetwork():

    def __init__(self, input_shape):
        self.hidden_sizes = [2*input_shape[1], 2*input_shape[1], input_shape[1], input_shape[1]/2]
        net = tflearn.input_data (shape=input_shape)
        for nhidden in self.hidden_sizes:
            print ('tfnetwork:: making {} hidden layer'.format(nhidden))
            net = tflearn.fully_connected(net, nhidden, activation='relu')
        # final output layer
        net = tflearn.fully_connected(net, 1)

        self.nepochs = 10
        self.batch_size = 64
        self.qnet = tflearn.regression(net, loss='mean_square', batch_size=self.batch_size)
        print ('tfnetwork:: self.net = ', self.qnet)

        self.dnn = tflearn.DNN(self.qnet, tensorboard_verbose=0)
        print ('tfnetwork:: dnn= ', self.dnn)
    #

    def predict(self, X): # X = [ [state_one_hot, action_one_hot] ]
        #print ('predict(X={})'.format(X))
        q = self.dnn.predict(X) # returns array or list of arrays
        return q
    #

    def learn(self, X, Y):
        print ('learn: X={} Y={}'.format(len(X), len(Y)))
        print('X=', np.array(X)[0:3,:])
        print('Y=', np.array(Y)[0:3,:])

        self.dnn.fit(X, Y, batch_size=len(X),
                     n_epoch=self.nepochs, show_metric=True, run_id='QfunctionModel')
        pass
    #
#

class Qfunction():
    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions
        self.input_shape = [None, nstates+nactions]
        self.model = tfnetwork(self.input_shape)
        self.xlist = []
        self.ylist = []
    #
    def flush(self):
        self.xlist = []
        self.ylist = []
    #
    def _net_input(self, state, action):
        state_vec = onehot(self.nstates, state)
        action_vec = onehot(self.nactions, action)
        input_vec = np.concatenate([state_vec, action_vec])
        return input_vec
    #
    def predict(self, state, action):
        input_vec = self._net_input(state, action)
        input = np.array([input_vec]) # convert to 1xN matrix
        q = self.model.predict(input)
        #print ('predict: type(q)=', type(q), ' q=', q)
        return q[0][0]
    #
    def push(self, state, action, qtarget):
        input_vec = self._net_input(state, action)
        target_vec = np.array([qtarget])
        self.xlist.append(input_vec)
        self.ylist.append(target_vec)
        return
    #
    def learn(self):
        '''
        X = np.array(self.xbuffer)
        Y = np.array(self.ybuffer)
        self.model.learn(X, Y)
        '''
        X = self.xlist
        Y = self.ylist
        self.model.learn(X, Y)
        return
    #

    def eGreedy(self, env, state, eps=0.1):
        action = env.action_space.sample()
        if (self.model is not None) and (np.random.uniform() > eps):
            qs = self.predict(state, action)
            action = np.argmax(qs)
            #
        return action
        #
    #
# // end Qfunction


env = gym.make('FrozenLake-v44')

# make TF network model for approximating Q function
qnetwork = Qfunction(nstates=env.observation_space.n, nactions=env.action_space.n)

np.random.seed(7)
env.seed(7)
reward = 0
done = False
gamma = 0.99

episode_count = 10000
for ep in range (episode_count):
    state = env.reset() # ob denotes the state of the env
    action = qnetwork.eGreedy(env, state)
    env.render()
    step_count = 1
    while True:
        # carry out the action
        state2, reward, done, _ = env.step (action)
        env.render()
        # choose another action from stp1 (for SARSA)
        action2 = qnetwork.eGreedy(env, state2, 0.3 if ep<100 else 0.1)

        print('>>> {}/ep:{}='.format(step_count, ep),
              ' (s:{},a:{},r:{},d:{}; s\':{},a\'{})'.format(state, action, reward, done, state2, action2))

        # experiment: negative reinforcement!
        if done and not (reward>0):
            reward += -1

        targetValue = reward
        if not done:
            qvalue = qnetwork.predict (state2, action2)
            print (' -- qvalue = {} of type {}'.format(qvalue, type(qvalue)))
            targetValue += gamma * qvalue
        #

        # the target value of Q(state,action) is this targetValue
        qnetwork.push (state, action, targetValue)

        if done:
            if reward>0:
                print ('!!! final location visited !!!\n\n\n')

            print ('-- Episode finished. Q-Network will learn the data.')
            qnetwork.learn()
            qnetwork.flush()
            break

        # update variables
        state = state2
        action = action2
    # end while True
#
print ('finished.')
