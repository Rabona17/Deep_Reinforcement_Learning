import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses

class dqn(keras.Model):
    def __init__(self, n):
        super(dqn, self).__init__()
        self.layer1 = layers.Dense(64, activation='relu', kernel_initializer='he_normal', )
        self.layer2 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.layer3 = layers.Dense(n, kernel_initializer='he_normal')

    def call(self, x, training=None):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
