import gym
import numpy as np
import tensorflow as tf

def onehot(x):
    '''
    :param x:
    :return: a 1x16 matrix whose x-th elem. is 1
    '''
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v0')
input_size = env.observation_space.n
output_size = env.action_space.n

# make TF network for approximating Q function

tf.reset_default_graph()
X = tf.placeholder (shape=[1,input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

Qout = tf.matmul (X, W) # output of Q prediction
maxQ = tf.argmax (Qout, 1)

Y = tf.placeholder (shape=[1,output_size], dtype=tf.float32)
loss = tf.reduce_sum (tf.square (Y-Qout))

learning_rate = 0.01
trainer = tf.train.GradientDescentOptimizer (learning_rate=learning_rate)
updateQmodel = trainer.minimize(loss=loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

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

episode_count = 100

env.seed(0)
env.reset()
reward = 0
done = False
eps = 0.1    # epsilon-greedy
gamma = 0.99

for i in range (episode_count):
    state = env.reset() # ob denotes the state of the env
    step_count = 1
    while True:
        # action selection
        if np.random.uniform() < eps:
            action = env.action_space.sample()
        else:
            predict = sess.run (Qout, feed_dict={X: onehot(state)})
            action = np.argmax(predict)

        # do action
        nxtstate, reward, done, _ = env.step (action)

        print('{}/i{}='.format(step_count, i),
              #' predict=', predict,
              ' a=', action,
              ' ob=', nxtstate,
              ' reward=', reward, ' done=', done)

        y = np.zeros((1,output_size))
        if done:
            y[0,action] = reward
        else:
            predict = sess.run (Qout, feed_dict={X: onehot(nxtstate)})
            y[0,action] = reward + gamma * np.max(predict)

        x = onehot(state)
        sess.run(updateQmodel, feed_dict={X:x, Y:y})


        if done:
            break

        state = nxtstate
        step_count += 1
