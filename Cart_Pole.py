import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses
import matplotlib.pyplot as plt
import gym
import pandas as pd
import random
import collections

from dqn import dqn
from utils import *
from replay_buffer import replay_buffer
tf.random.set_seed(1234)
def main():
    optimizer = optimizers.Adam(lr=0.02)
    score_arr = np.array([])
    t_ = np.array([])
    env = gym.make('CartPole-v1')
    env.seed(1234)
    q = dqn()
    q_target = dqn()
    q.build(input_shape=(None,4))
    q_target.build(input_shape=(None,4))
    memory = replay_buffer()
    copy_round = 20
    score = 0.0
    for src, dest in zip(q.variables, q_target.variables):
        dest.assign(src)
    for n_epi in range(1, 10001):
        s = env.reset()
        for t in range(1200):
            a = epi_greedy(max(0.1, 0.4 - 0.01 * (n_epi / 200)), s,q,env.action_space.n)
            s_prime, r, done, info = env.step(a)
            #MODIFIED REWARD
            x, x_dot, theta, theta_dot = s_prime
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            r = r1 + r2
            done_mask = int(not done)
            memory.add_to_buffer(s, r / 100.0, a, s_prime, done_mask)
            s = s_prime
            score += r
            if done:
                break
        if memory.size() > 2000:
            train(memory,q, q_target, 0.99,32,optimizer)
        if n_epi % copy_round == 0:
            for src, dest in zip(q.variables, q_target.variables):
                dest.assign(src)
            print(n_epi, score/copy_round)
            t_ = np.append(t_, n_epi)
            score_arr = np.append(score_arr, score/copy_round)
            score = 0.0
            
            
    env.close()
    plt.plot(t_, score_arr)
    plt.show()

if __name__ == '__main__':
    main()
