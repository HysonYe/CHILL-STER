import numpy as np
from uwan.utils import unpack_place_info, get_delay_slot
from uwan.mac.base_mac import BaseMAC
from mobility import AUVMobilityRMP
from infra.environment import COORD_ORDER

class Node:
    def __init__(self, id, env_configs, node_configs, mobility_model:AUVMobilityRMP = None):
        self.id = id
        self.type = 'NODE'

        # Environment attributes
        self._space = env_configs['space']
        self._prop_speed = env_configs['prop_speed']
        self._slot_size = env_configs['slot_size']
        self._AP_pos = env_configs['AP_pos']
        
        # Node attributes
        self._place_strategy = node_configs['place_strategy']
        self.mobility_model = mobility_model
        self.delay = -1
        self.pos = {}
        self._t = 0
        
        # Protocol interface
        self.mac_protocol = None

    def attach_protocol(self, protocol: BaseMAC):
        # Bind the protocol to the node
        self.mac_protocol = protocol
        self.mac_protocol.attach_to_node(self)
        self.type = protocol.protocol_type

    def _apply_scheduled_pos(self):
        if str(self._t) in self._place_strategy.keys():
            place_info = self._place_strategy[str(self._t)]
            self.delay, self.pos = unpack_place_info(
                place_info, self._slot_size, self._prop_speed, self._AP_pos, self._space
            )
            if self.mobility_model:
                coord = [self.pos[xyz] for xyz in COORD_ORDER]
                self.mobility_model.set_position(coord)

    def _auv_step(self):
        if self.mobility_model:
            self.mobility_model.step(self._slot_size)
            self.delay = get_delay_slot(
                self.mobility_model.get_position(),
                self._AP_pos,
                self._prop_speed,
                self._slot_size
            )
            current_pos = self.mobility_model.get_position()
            self.pos = {COORD_ORDER[i]: current_pos[i] for i in range(len(COORD_ORDER))}

    def reset(self):
        self._t = 0
        self._apply_scheduled_pos()
        if self.mobility_model:
            self.mobility_model._initial_pos = np.array(self.mobility_model.get_position(), dtype=float)
            self.mobility_model.reset()
        
        if self.mac_protocol:
            self.mac_protocol.reset()
    
    def mac_decide(self):
        access_decision = 0
        send_packet = None
        if self.mac_protocol:
            access_decision = self.mac_protocol.decide()
        if access_decision == 1:
            send_packet = {
                'src': self.id,
                'dst': 'AP',
                'pos': self.pos,
                'delay': self.delay
            }
        return access_decision, send_packet

    def step(self, downlink_packets=None):
        self._auv_step()            # Update position and delay based on mobility model
        ack_frame = None
        if self.mac_protocol:       # Execute protocol logic and obtain feedback/ACK frames
            ack_frame = self.mac_protocol.step(downlink_packets)
        self._t += 1
        self._apply_scheduled_pos() # Update position and delay based on deployment strategy
        return ack_frame