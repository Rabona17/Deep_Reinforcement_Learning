import random
import collections
class replay_buffer():
    def __init__(self, max_size=50000):
        self.buffer = collections.deque(maxlen=max_size)

    def add_to_buffer(self, s, r, a, s_, done):
        self.buffer.append((s, r, a, s_, done))

    def get_random(self, n):
        return random.sample(self.buffer, n)

    def size(self):
        return len(self.buffer)
