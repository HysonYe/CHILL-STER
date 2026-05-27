from uwan.mac.base_mac import BaseMAC
import copy

class TDMA(BaseMAC):
    def __init__(self, slot_strategy):
        super().__init__(protocol_type='TDMA')
        self._slot_strategy = slot_strategy
        self._strategy_request = None
        self._sent_slot = None

    def _update_strategy(self, strategy):
        if len(strategy['fixed']) > 0:
            self._sent_slot = strategy['fixed']
            _slot_type = 'fixed'
            print('[Info] TDMA(id = {}) update {}-strategy = {}, at time step {}.'.format(
                self._node.id, _slot_type, self._sent_slot, self._node._t))
        else:
            self._strategy_request = strategy['rand']

    def get_strategy_request(self):
        if self._strategy_request is None: return None
        ret = copy.deepcopy(self._strategy_request)
        self._strategy_request = None
        return ret

    def set_strategy(self, strategy):
        self._sent_slot = strategy
        _slot_type = 'rand'
        print('[Info] TDMA(id = {}) update {}-strategy = {}, at time step {}.'.format(
            self._node.id, _slot_type, self._sent_slot, self._node._t))

    def reset(self):
        self._strategy_request = None
        self._update_strategy(self._slot_strategy[str(self._node._t)])
    
    def decide(self):
        slots = self._node._t
        idx = slots % len(self._sent_slot)
        act = self._sent_slot[idx]
        return act

    def step(self, downlink_packets=None):
        next_t = self._node._t + 1
        if str(next_t) in self._slot_strategy.keys():
            new_strategy = self._slot_strategy[str(next_t)]
            self._update_strategy(new_strategy)