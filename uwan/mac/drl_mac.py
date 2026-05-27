from uwan.mac.base_mac import BaseMAC
from infra.environment import ACK_STATE
class DRLMAC(BaseMAC):
    def __init__(self):
        super().__init__(protocol_type='AGENT')
        self._next_action = 0

    def set_action(self, action):
        self._next_action = action

    def _get_ack_frame(self, downlink_packets):
        ack_frame = {'ack': ACK_STATE['NO_ACK']}
        log, idx, size = downlink_packets['log'], downlink_packets['idx'], downlink_packets['max']
        if len(log) == 0: return ack_frame
        if len(log) >= size:
            rec_idx = (idx - self._node.delay) % size
        else:
            rec_idx = max(idx - self._node.delay, 0)
            
        ap_data = log[rec_idx]
        addr, ack = ap_data['addr'], ap_data['data']
        if addr != -1:
            ack_frame = {
                'ack': ACK_STATE['ACK'] if addr == 0 else ACK_STATE['OTHER_ACK'],
            }
        return ack_frame

    def reset(self):
        self._next_action = 0
    
    def decide(self):
        return self._next_action

    def step(self, downlink_packets=None):
        ack_frame = self._get_ack_frame(downlink_packets)
        return ack_frame