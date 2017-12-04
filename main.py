import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


env = gym.make('MountainCar-v0')

# Fetching action space and observation space size
action_space_size = env.action_space.n
observation_space_size = env.observation_space.shape

# Dimensions of output at each layer
d1 = 20
d2 = 18
d3 = 10

n = observation_space_size[0]

# Regularization
gamma_r = 0.01


# Function Approximator Consisting of Neural Network
inputs = tf.placeholder(tf.float32, (None, env.observation_space.shape[0]), name='Input')
targets = tf.placeholder(tf.float32, (None, action_space_size), name='Target')
w1 = tf.Variable(tf.random_normal([observation_space_size[0], d1]), name='w1')
w2 = tf.Variable(tf.random_normal([d1, d2]) * tf.cast(tf.sqrt(1/tf.pow(n, 1)), tf.float32), name='w2')
w3 = tf.Variable(tf.random_normal([d2, d3]) * tf.cast(tf.sqrt(1/tf.pow(n, 2)), tf.float32), name='w3')
w4 = tf.Variable(tf.random_normal([d3, action_space_size]) * tf.cast(tf.sqrt(1/tf.pow(n, 3)), tf.float32), name='w4')

b1 = tf.Variable(tf.zeros([d1]), name='b1')
b2 = tf.Variable(tf.zeros([d2]), name = 'b2')
b3 = tf.Variable(tf.zeros([d3]), name = 'b3')
b4 = tf.Variable(tf.zeros([action_space_size]), name='b4')


o1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
o2 = tf.nn.relu(tf.matmul(o1, w2) + b2)
o3 = tf.nn.relu(tf.matmul(o2, w3) + b3)
o4 = tf.nn.softmax(tf.matmul(o3, w4) + b4)


init = tf.global_variables_initializer()


def w_sum(W):
    tot_sum = 0
    for w in W:
        temp = tf.squeeze(w)
        tot_sum += tf.reduce_sum(temp)

    return tot_sum


penalty = tf.divide(tf.multiply(gamma_r, tf.cast(w_sum([w1, w2, w3, w4]), tf.float32)), 2.0 * tf.cast(tf.shape(targets)[0], tf.float32))

error = tf.reduce_mean(tf.square(targets - o4), axis=0) + penalty

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001, name='Gradient_Descent_Optimizer').minimize(error)

# Gain Experience
e = 3000                                                                       # Number of Episodes
s = 500                                                                      # Number of steps
gamma = 0.1                                                                # Reward Discount

loss_func = []
reward_func = []
print('Learning Begins')
with tf.Session() as sess:
    sess.run(init)
    for episode in range(0, e):
        if episode%1000 == 0:
            print('o')
        obs = env.reset()
        memory = deque()

        # Initiating An Episode
        for step in range(0, s):

            # Selecting Action
            action = env.action_space.sample()

            # Performing Action
            obs_next, reward, done, info = env.step(action)

            # Registering Experience in Memory
            memory.append((obs, obs_next, action, reward, done))

            # Updating the state
            if done:
                obs = env.reset()
                break
            else:
                obs = obs_next

        # Experience Replay
        # Declaring variables for features and labels
        features = []
        labels = []

        # Traversing the batch
        for experience in memory:
            state = experience[0]
            next_state = experience[1]
            action = experience[2]
            reward = experience[3]
            termination = experience[4]

            if not termination:
                total_reward = reward + gamma* np.max(sess.run(o4, feed_dict={inputs:np.reshape(state, (1,observation_space_size[0]))}))
            else:
                total_reward = reward

            action_q = np.zeros((action_space_size))
            action_q[action] = total_reward

            features.append(state)
            labels.append(action_q)


        # Training Neural Network based on generated features and labels
        _, loss = sess.run([train_step, error], feed_dict={inputs: features, targets:labels})

        loss_func.append(loss)

    loss_np = np.mean(np.array(loss_func), axis=1)

    plt.plot(loss_np, color = 'green')
    plt.show()

    save_path = tf.train.Saver().save(sess, save_path="tmp/model1.ckpt")

