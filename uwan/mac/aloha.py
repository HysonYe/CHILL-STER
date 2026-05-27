from uwan.mac.base_mac import BaseMAC
import numpy as np

class ALOHA(BaseMAC):
    def __init__(self, prob_q_strategy):
        super().__init__(protocol_type='ALOHA')
        self._prob_q_strategy = prob_q_strategy
        self._prob_q = None

    def reset(self):
        self._prob_q = self._prob_q_strategy[str(self._node._t)]

    def decide(self):
        prob = np.random.uniform(0, 1)
        act = 1 if prob <= self._prob_q else 0
        return act

    def step(self, downlink_packets=None):
        next_t = self._node._t + 1
        if str(next_t) in self._prob_q_strategy.keys():
            self._prob_q = self._prob_q_strategy[str(next_t)]
