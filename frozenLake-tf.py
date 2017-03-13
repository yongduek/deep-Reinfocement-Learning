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

def eGreedy(env, st, qnetwork, eps=0.1):
    action = env.action_space.sample()
    if (qnetwork is not None) and (np.random.uniform() > eps):
        qs = qnetwork.predict(st)
        action = np.argmax(qs)
#
    return action
#

def onehot(x):
    '''
    :param x:
    :return: a 1x16 matrix whose x-th elem. is 1
    '''
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v44')
input_size = env.observation_space.n
output_size = env.action_space.n

# make TF network model for approximating Q function

'''
   Deep SARSA-learning
   SARSA network: input=state/observation, output = Q value of actions
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

qnetwork = None


env.seed(0)
reward = 0
done = False
gamma = 0.99

episode_count = 1
for i in range (episode_count):
    state = env.reset() # ob denotes the state of the env
    action = eGreedy(env, state, qnetwork)
    env.render()
    step_count = 1
    while True:
        # carry out the action
        stp1, reward, done, _ = env.step (action)
        env.render()

        # choose another action from stp1 (for SARSA)
        action2 =  eGreedy(env, stp1, qnetwork)

        print('{}/i{}='.format(step_count, i),
              ' a=', action,
              ' stp1=', stp1,
              ' reward=', reward, ' done=', done)

        targetValue = reward
        if not done:
            qvalue = qnetwork.predict (stp1, action2)
            targetValue += gamma * qvalue

        # the target value of Q(state,action) is this targetValue
        # so, update Q network with learning rate alpha

        # 
        state = stp1
        action = action2

        if done:
            break
#
print ('finished.')
