from agents.base_agent import BaseAgent
from infra.environment import OBS_ORDER
from infra.utils.tools import up_to_integer
from collections import deque
import numpy as np

class MobiUMAC(BaseAgent):
    '''
    MobiU-MAC protocol
    '''
    def __init__(self, gamma):
        super().__init__(gamma)
    
    def reset(self):
        self.t = 0

    def update_delay(self, observation = None, ref_delay = -1):
        pass

    def build_state(self, observation = None):
        pass
    
    def store_transition(self, state, action, reward, next_state, prob, observation):
        pass

    def choose_action(self, states):
        pass

    def save_checkpoint(self, path):
        pass

    def learn(self):
        pass

class ImplicitAnchorEstimator:

    def __init__(self, max_offset, window_size,
                 node_act=None, ack_state=None, obs_order:list =None):
        # copy params
        self._max_offset = max_offset
        self._window_size = window_size
        self._node_act  = node_act
        self._ack_state = ack_state
        self._act_idx = obs_order.index('act')
        self._ack_idx = obs_order.index('ack')

        # internal states
        self._steps = window_size - max_offset
        self._buf = deque(maxlen=window_size)
        self._offset_deque = deque(maxlen=20)
        self._offset_acc = 0
        self._scores = np.full(max_offset + 1, 0, dtype=np.int32)
        self._scores[:2] = -10**9
        self._init_flag = False

    def _compute_score(self, act, ack):
        if act == self._node_act['SEND']:
            if   ack == self._ack_state['ACK']:        return  1
            elif ack == self._ack_state['OTHER_ACK']:  return -1
            else: return 0
        elif act == self._node_act['WAIT']:
            if   ack == self._ack_state['NO_ACK']:     return  1
            elif ack == self._ack_state['ACK']:        return -1
            else: return 0
    
    def _smooth_offset(self, optimal_offset):
        self._offset_acc += optimal_offset
        if len(self._offset_deque) == self._offset_deque.maxlen:
            self._offset_acc -= self._offset_deque[0]
        self._offset_deque.append(optimal_offset)
        return self._offset_acc / len(self._offset_deque)

    def push(self, obs):
        '''
        Pushes a new observation and returns the estimated implicit spatial anchor z_t.
        Returns None if the sliding window is not yet full.
        '''
        buf, W, R = self._buf, self._window_size, self._max_offset + 1

        # step 1. Remove old observation and deduct the corresponding score
        if len(buf) == W:
            old_act = buf[0][self._act_idx]
            for offset in range(2, R):
                old_ack = buf[offset][self._ack_idx]
                self._scores[offset] -= self._compute_score(old_act, old_ack)

        # step 2. Add new observation and add the corresponding score
        buf.append(obs)
        if len(buf) < W: return None
        
        if self._init_flag == False:
            # Initialization
            self._init_flag = True
            for offset in range(2, R):
                for t in range(self._steps):
                    self._scores[offset] += self._compute_score(buf[t][self._act_idx], buf[t + offset][self._ack_idx])
        else:
            t = self._steps - 1
            act_in = buf[t][self._act_idx]
            for offset in range(2, R):
                ack_in = buf[t + offset][self._ack_idx]
                self._scores[offset] += self._compute_score(act_in, ack_in)

        # step 3. Select optimal offset and apply smoothing to get final implicit spatial anchor estimate
        best_offset = int(np.argmax(self._scores))
        smoothed_m = self._smooth_offset(best_offset)
        isa = up_to_integer(smoothed_m / 2)
        return isa