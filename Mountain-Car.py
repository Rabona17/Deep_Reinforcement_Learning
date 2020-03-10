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
    env = gym.make('MountainCar-v0')
    env.seed(1234)
    q = dqn(env.action_space.n)
    q_target = dqn(env.action_space.n)
    q.build(input_shape=(None,env.observation_space.shape[0]))
    q_target.build(input_shape=(None,env.observation_space.shape[0]))
    memory = replay_buffer()
    copy_round = 20
    score = 0.0
    for src, dest in zip(q.variables, q_target.variables):
        dest.assign(src)
    for n_epi in range(1, 3000):
        s = env.reset()
        for t in range(1200):
            if(n_epi>1500):
                env.render()
            a = epi_greedy(max(0.1, 0.4 - 0.01 * (n_epi / 200)), s,q,env.action_space.n)
            #print(a)
            s_prime, r, done, info = env.step(a)
            #MODIFIED REWARD
            pos, vel = s_prime
            r = 3*abs(pos+0.5)
            done_mask = int(not done)
            memory.add_to_buffer(s, r , a, s_prime, done_mask)
            s = s_prime
            score += r
            if done:
                break
        if memory.size() > 5000:
            train(memory,q, q_target, 0.99,64,optimizer)

        if n_epi % 60 == 0:
            for src, dest in zip(q.variables, q_target.variables):
                dest.assign(src)
        if n_epi % copy_round == 0:
            print(n_epi, score/copy_round)
            t_ = np.append(t_, n_epi)
            score_arr = np.append(score_arr, score/copy_round/3)
            score = 0.0
            
            
    env.close()
    plt.plot(t_, score_arr)
    plt.show()

if __name__ == '__main__':
    main()
