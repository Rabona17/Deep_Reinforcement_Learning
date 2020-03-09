import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses

def epi_greedy(epi, s, dqn, a_len):
    s = tf.constant(s.reshape([1,-1]),tf.float32)
    out = dqn(s)[0].numpy().argmax()
    if random.random() > epi:
        return out
    else:
        return random.randint(0, a_len-1)

def train(buffer, q_eval, q_target, gamma, batch, optimizer):
    huber = losses.Huber()
    samp = buffer.get_random(batch)
    sl, rl, al, s_l, dl = [], [], [], [], []
    def add(i):
        sl.append(i[0])
        rl.append(i[1])
        al.append(i[2])
        s_l.append(i[3])
        dl.append(i[4])
    list(map(add,samp))
    s = tf.constant(np.array(sl), dtype=tf.float32)
    r = tf.constant(np.array(rl).reshape([-1,1]),dtype=tf.float32)
    s_ = tf.constant(np.array(s_l),dtype=tf.float32)
    d = tf.constant(np.array(dl).reshape([-1,1]),dtype=tf.float32)
    with tf.GradientTape() as tape:
        q_out = q_eval(s)
        idx = tf.constant(np.asarray(list(zip(range(len(al)), al))),dtype=tf.int32)
        q_a = tf.reshape(tf.gather_nd(q_out, idx),[-1,1])
        q_optima  = tf.reduce_max(q_target(s_),axis=1,keepdims=True)
        q_optima = r + gamma * q_optima  * d
        loss = huber(q_a, q_optima)
    grads = tape.gradient(loss, q_eval.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_eval.trainable_variables))
