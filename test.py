import gym
import tensorflow as tf
import numpy as np


env = gym.make('Acrobot-v1')

# Fetching action space and observation space size
action_space_size = env.action_space.n
observation_space_size = env.observation_space.shape

# Dimensions of output at each layer
d1 = 20
d2 = 18
d3 = 10

# Function Approximator Consisting of Neural Network

inputs = tf.placeholder(tf.float32, (None, env.observation_space.shape[0]), name='Input')
targets = tf.placeholder(tf.float32, (None, action_space_size), name='Target')
w1 = tf.Variable(tf.zeros([observation_space_size[0], d1]), name='w1')
w2 = tf.Variable(tf.zeros([d1, d2]), name='w2')
w3 = tf.Variable(tf.zeros([d2, d3]), name='w3')
w4 = tf.Variable(tf.zeros([d3, action_space_size]), name='w4')

b1 = tf.Variable(tf.zeros([d1]), name='b1')
b2 = tf.Variable(tf.zeros([d2]), name = 'b2')
b3 = tf.Variable(tf.zeros([d3]), name = 'b3')
b4 = tf.Variable(tf.zeros([action_space_size]), name='b4')

o1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
o2 = tf.nn.relu(tf.matmul(o1, w2) + b2)
o3 = tf.nn.relu(tf.matmul(o2, w3) + b3)
o4 = tf.nn.softmax(tf.matmul(o3, w4) + b4)

with tf.Session() as sess:
    tf.train.Saver().restore(sess, "tmp/model1.ckpt")

    writer = tf.summary.FileWriter("output", graph=tf.get_default_graph())
    # Testing our model
    print('Game Begins')
    test_obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        env.render()
        Q = sess.run(o4, feed_dict={inputs: np.reshape(test_obs, (1,observation_space_size[0]))})

    # Selecting action with maximum Q value
        action = np.argmax(Q)

    # Performing Action
        obs_next, reward, done, info = env.step(action)

        test_obs = obs_next
        total_reward += reward
    writer.close()
    print('Game Over! Total Reward: ', total_reward)