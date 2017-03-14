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

class tfnetwork():
    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions
        self.input_shape = [None, nstates+nactions]
        self.hidden_sizes = [2*self.input_shape[1], 2*self.input_shape[1], nactions]
        net = tflearn.input_data (shape=self.input_shape)
        for nhidden in self.hidden_sizes:
            print ('tfnetwork:: making {} hidden layer'.format(nhidden))
            net = tflearn.fully_connected(net, nhidden, activation='relu')
        # final output layer
        net = tflearn.fully_connected(1)

        self.nepochs = 10
        self.batch_size = 64
        self.qnet = tflearn.regression(net, loss='mean_square', self.batch_size)
        print ('tfnetwork:: self.net = ', self.net)

        self.dnn = tflearn.DNN(self.qnet, tensorboard_verbose=0)
    #

    def predict(self, state, action):
        q = self.dnn.predict(np.concatenate([state, action]))
        return q
    #
    def learn(self, X, Y):
        self.dnn.fit(X, Y, n_epoch=self.nepochs, show_metric=True, run_id='QfunctionModel')
        pass
    #
#

class Qfunction():
    def __init__(self, nstates, nactions):
        self.model = tfnetwork(nstates, nactions)
        self.xbuffer = []
        self.ybuffer = []
    #

    def predict(self, state, action):
        q = self.model.predict(state, action)
        return q
    #
    def push(self, state, action, qtarget):
        state_action = np.concatenate([state,action])
        self.xbuffer.append(state_action)
        self.ybuffer.append(qtarget)
        return
    #
    def learn(self):
        X = np.array(self.xbuffer)
        Y = np.array(self.ybuffer)
        self.model.learn(X,Y)
        return
    #

    def eGreedy(self, env, state, eps=0.1):
        action = env.action_space.sample()
        if (self.model is not None) and (np.random.uniform() > eps):
            qs = self.model.predict(state, action)
            action = np.argmax(qs)
            #
        return action
        #


#


def onehot(x):
    '''
    :param x:
    :return: a 1x16 matrix whose x-th elem. is 1
    '''
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v44')

# make TF network model for approximating Q function
qnetwork = Qfunction(nstates=env.observation_space.n, nactions=env.action_space.n)


env.seed(0)
reward = 0
done = False
gamma = 0.99

episode_count = 1
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
        action2 = qnetwork.eGreedy(env, state2)

        print('>>> {}/ep:{}='.format(step_count, ep), ' ({},{},{},{}; {},{})'.format(state, action, reward, done, state2, action2))

        targetValue = reward
        if not done:
            qvalue = qnetwork.predict (state2, action2)
            targetValue += gamma * qvalue
        #

        # the target value of Q(state,action) is this targetValue
        qnetwork.push ([state, action, targetValue])
        # 
        state = state2
        action = action2

        if done:
            qnetwork.learn()
            break
#
print ('finished.')
