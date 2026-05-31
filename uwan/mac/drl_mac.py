from uwan.mac.base_mac import BaseMAC
from infra.environment import ACK_STATE, MacReturn, AckFrameInfo
class DRLMAC(BaseMAC):
    def __init__(self):
        super().__init__(protocol_type='AGENT')
        self._beacons = {}
        self._next_action = 0
        self._measured_delay = -1

    def reset(self):
        self._next_action = 0
    
    def set_action(self, action):
        self._next_action = action
        
    def measure_delay(self):
        measured_delay = self._node.delay
        time_slot = self._node._t
        self._beacons[time_slot + 2*measured_delay] = measured_delay
    
    @property
    def measured_delay(self):
        time_slot = self._node._t
        if time_slot in self._beacons:
            new_delay = self._beacons.pop(time_slot)
            self._measured_delay = new_delay
            print(f'[Info] Agent receives beacon at steps {time_slot}, measured delay updated to {new_delay}.')
        return self._measured_delay

    def decide(self):
        return self._next_action

    def step(self, downlink_packets=None) -> MacReturn:
        ack_frame = self._get_ack_frame(downlink_packets)
        ret = {
            'ack_frame': ack_frame,
            'measured_delay': self.measured_delay
        }
        return ret
    
    def _get_ack_frame(self, downlink_packets) -> AckFrameInfo:
        ack_frame: AckFrameInfo = {'state': ACK_STATE['NO_ACK']}
        log, idx, size = downlink_packets['log'], downlink_packets['idx'], downlink_packets['max']
        if len(log) == 0: return ack_frame
        if len(log) >= size:
            rec_idx = (idx - self._node.delay) % size
        else:
            rec_idx = max(idx - self._node.delay, 0)
            
        ap_data = log[rec_idx]
        addr, ack = ap_data['addr'], ap_data['data']
        if addr != -1:
            ack_frame['state'] = ACK_STATE['ACK'] if addr == 0 else ACK_STATE['OTHER_ACK']
        return ack_frame