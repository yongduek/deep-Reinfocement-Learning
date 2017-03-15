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

        self.nepochs = 100
        self.batch_size = 64
        self.qnet = tflearn.regression(net, loss='mean_square', batch_size=self.batch_size)
        print ('tfnetwork:: self.net = ', self.qnet)

        self.dnn = tflearn.DNN(self.qnet, tensorboard_verbose=-1)
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

        self.dnn.fit(X, Y,
                     batch_size=len(X),
                     n_epoch=self.nepochs,
                     show_metric=False,
                     run_id='QfunctionModel')
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
        state_vec = state
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

    def eGreedy(self, state, eps=0.1):
        a = None # action
        u = np.random.uniform()
        if (self.model is not None) and (u < eps):
            a = np.random.randint(0, self.nactions)
            #print ('eGreedy: eps={} action = {} by u={}'.format(eps, a, u))
        else:
            qvalues = [self.predict(state, a) + 0.001*np.random.uniform()
                       for a in range(self.nactions)]
            a = np.argmax(qvalues)
            #print ('eGreedy: qvalues={} and action = {}'.format(qvalues, a))
        #
        return a
        #
    #
# // end Qfunction


#env = gym.make('FrozenLake-v44')
env = gym.make('CartPole-v0')
is_statespace_box = type(env.observation_space)==gym.spaces.box.Box
is_actionspace_box = type(env.action_space)==gym.spaces.box.Box

# make TF network model for approximating Q function
nstates = env.observation_space.shape[0] if is_statespace_box \
                                        else env.observation_space.n
nactions = env.actino_space.shape[0] if is_actionspace_box \
                                        else env.action_space.n
#
qnetwork = Qfunction(nstates, nactions)

# ----------------------------
np.random.seed(7)
env.seed(7)
reward = 0
done = False
gamma = 0.999

is_goal_reached = False
max_episode_count = 10000
for ep in range (max_episode_count):

    state = env.reset() # ob denotes the state of the env
    action = qnetwork.eGreedy(state)
    env.render()
    reward_sum = 0
    step_count = 0
    while True:
        step_count += 1

        # carry out the action
        state2, reward, done, _ = env.step (action)
        env.render()
        reward_sum += reward

        # choose another action from stp1 (for SARSA)
        action2 = qnetwork.eGreedy(state2, 0.05)
        #action2 = qnetwork.eGreedy(state2, 0.5 if not is_goal_reached else 0.1)

#        print('>>> {}/ep:{}='.format(step_count, ep),
#              ' (s:{},a:{},r:{},d:{}; s\':{},a\'{})'.format(state, action, reward, done, state2, action2))

        targetValue = reward
        if not done:
            qvalue = qnetwork.predict (state2, action2)
            #print(' -- qvalue = {} of type {}'.format(qvalue, type(qvalue)))
            targetValue += gamma * qvalue
        #

        # the target value of Q(state,action) is this targetValue
        qnetwork.push(state, action, targetValue)

        if done:
            print('-- Episode {} ended in {} steps : reward sum = {}'.format(ep, step_count, reward_sum))
            qnetwork.learn()
            qnetwork.flush()
            break

        # update variables
        state = state2
        action = action2
    # end while True
#
print ('finished.')
