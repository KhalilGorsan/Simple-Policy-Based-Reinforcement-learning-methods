import numpy as np


class Policy:
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4 * np.random.rand(
            s_size, a_size
        )  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)  # option 2: deterministic policy
        return action