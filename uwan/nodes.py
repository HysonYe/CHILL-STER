from uwan.utils import unpack_place_info, get_delay_slot
from uwan.mac.base_mac import BaseMAC
from mobility import AUVMobilityRMP
from infra.environment import COORD_ORDER, AP_COORD
import numpy as np
import copy

class Node:
    '''
    User Node, AUV Node
    '''
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

class AccessPoint():
    '''
    AP Node
    '''
    
    AP_STATE = {'vacant': 0, 'successful': 1, 'collided': 0}
    # NOTE: Considering realistic scenarios, we assume the AP treats collisions 
    # and idle slots identically and returns no ACK for either case.
    def __init__(self, nodes_num):
        self.pos = {}
        self._nodes_num = nodes_num

    def reset(self):
        for i, xyz in enumerate(COORD_ORDER):
            self.pos[xyz] = AP_COORD[i]
        self._AP_receive = {}
        self._AP_sent = {
            'idx': 0,
            'max': 200,
            'log': []
        }
        self._t = 0

    def _receive(self, id, delay):
        slot = self._t + delay
        if slot not in self._AP_receive.keys():
            self._AP_receive[slot] = [0] * self._nodes_num
        self._AP_receive[slot][id] += 1 # Explicitly record the number of packets received by the AP from each node in a specific slot
    
    def _sent(self, node_id = -1, receive_result = 0):
        '''
        Feedback on the reception result of data frames (return ACK)

        Args:
            node_id (int): ID of the sending node, -1 indicates no actual feedback sent (e.g., during collision or idle slots)
            receive_result (int): Reception result of the AP in the current slot, which can be:
                - AP_STATE['vacant'] (0): No data frames received by the AP in the current slot
                - AP_STATE['successful'] (1): One data frame successfully received by the AP in the current slot
                - AP_STATE['collided'] (0): Multiple data frames received by the AP, resulting in a collision
        '''
        idx = self._AP_sent['idx']
        log = self._AP_sent['log']
        size = self._AP_sent['max']
        self._AP_sent['idx'] = (idx + 1) % size # Update idx
        if receive_result == self.AP_STATE['successful']:
            # If the AP successfully receives a data frame, record the sender ID and send an ACK
            sent_data = {
                'addr': node_id,
                'data': receive_result
            }
        else:
            # Record a special identifier (e.g., -1) in case of collision/idle to indicate no actual ACK was sent
            sent_data = {
                'addr': -1,
                'data': receive_result
            }
        if len(log) < size:
            log.append(sent_data)
        else:
            log[idx] = sent_data

    def step(self, packets):
        '''
        Step function, processing received data frames and returning downlink data (ACK feedback)

        Args:
            packets (list[dict]): List of dictionaries containing sender ID and delay for each data frame
                - id (int): ID of the sending node
                - delay (int): Delay between the sender and the AP
        Returns:
            downlink_packets (dict): Downlink data returned by the AP, containing reception results
                - idx (int): Index of the current slot
                - max (int): Maximum length of the log
                - log (list[int]): Reception result log, containing results for each slot
        '''
        for packet in packets:                      # Receive packet
            self._receive(packet['id'], packet['delay'])

        if self._t in self._AP_receive.keys():      # Determine reception
            packets = self._AP_receive.pop(self._t)
            size = sum(packets)
            if size == 1:
                receive_result = self.AP_STATE['successful']
            else:
                receive_result = self.AP_STATE['collided']
        else:
            packets = [0] * self._nodes_num
            receive_result = self.AP_STATE['vacant']

        if receive_result == self.AP_STATE['successful']:   # Send reception result (ACK feedback)
            addr = packets.index(1)
            self._sent(node_id=addr, receive_result=receive_result)                  
        else:
            self._sent(receive_result=receive_result)
        self._t += 1
        downlink_packets = copy.deepcopy(self._AP_sent)
        return downlink_packets                             # Return downlink data