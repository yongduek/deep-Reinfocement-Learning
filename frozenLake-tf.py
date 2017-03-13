import gym
from gym.envs.registration import register
import numpy as np
import tensorflow as tf
import tflearn

register(
    id='FrozenLake-v44',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)
    #reward_threshold=0.78, # optimum = .8196

def onehot(x):
    '''
    :param x:
    :return: a 1x16 matrix whose x-th elem. is 1
    '''
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v0')
input_size = env.observation_space.n
output_size = env.action_space.n

# make TF network model for approximating Q function

Qmodel
'''
   Deep Q-learning
   Q network: input=state/observation, output = Q value of actions
   for each episode
        initialize
        for t=1, ..., T
            action = e-greedy {random, max_a Qpred}
            do_action, observe reward and next state

            y = reward             for terminal state
                reward + gamma max_a' Q(new_state, a' | theta)
            perform (y - Q(theta))^2 minimization
        endfor
    endfor
'''

qnetwork = None

episode_count = 100

env.seed(0)
reward = 0
done = False
gamma = 0.99

for i in range (episode_count):
    state = env.reset() # ob denotes the state of the env
    step_count = 1
    while True:
        def eGreedy(env, st, qnetwork, eps=0.1):
            action = env.action_space.sample()
            if np.random.uniform() > eps:
                qs = qnetwork.predict(st)
                action = np.argmax(qs)
            #
            return action
        #

        action = eGreedy(env, state, qnetwork)

        # do action
        stp1, reward, done, _ = env.step (action)

        print('{}/i{}='.format(step_count, i),
              ' a=', action,
              ' stp1=', stp1,
              ' reward=', reward, ' done=', done)


        if done:
            break
