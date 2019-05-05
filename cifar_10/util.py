import numpy as np


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.5, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    return schedule
