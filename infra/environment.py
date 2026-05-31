import copy
from typing import TypedDict
from uwan.utils import allocate_tdma_slots
from uwan.mobility import AUVMobilityRMP
from uwan.nodes import Node, AccessPoint
from uwan.mac import TDMA, ALOHA, DRLMAC

AP_COORD = [0.0, 0.0, 0.0]  # Default position of the Access Point
OBS_ORDER = ['act', 'ack']
COORD_ORDER = ['x', 'y', 'z']

NODE_ACT = {
    'WAIT': 0,
    'SEND': 1,
}

ACK_STATE = {
    'NO_ACK': 0,
    'ACK': 1,
    'OTHER_ACK': 2,  # ACK intended for another node
}

class AckFrameInfo(TypedDict, total=False):
    state: int  # ACK_STATE['NO_ACK'], ACK_STATE['ACK'], ACK_STATE['OTHER_ACK']

class MacReturn(TypedDict, total=False):
    ack_frame: AckFrameInfo
    measured_delay: int

class Environment():
    '''
    Underwater Acoustic Network MAC Simulation Environment.
    
    This class supports multiple normal nodes and a single AI agent node (extensible to multiple agents).
    All nodes share a single channel to communicate with the Access Point.
    '''

    def __init__(self, scene_config):
        '''
        Initialize the environment.

        Args:
            scene_config (dict): Configuration dictionary for the scenario.
                - environment (dict): Environment parameters including slot size, propagation speed, and space dimensions.
                - nodes (list[dict]): Parameters for normal nodes, including node type, placement strategy, and mac strategy.
                - agents (list[dict]): Parameters for agent nodes, including speed and placement strategy.
                - targets (dict): Target parameters, including goal positions.
        '''
        # environment parameters
        self._scene_config = scene_config
        self._env = self._scene_config['environment']
        self._slot_size = self._env['slot_size']
        self._prop_speed = self._env['prop_speed']
        self._space = self._env['space']
        self._targets = self._scene_config['targets']

        # action and observation space
        self._action_space = list(NODE_ACT.keys())
        self.n_actions = len(self._action_space)
        self.n_features = len(OBS_ORDER)
        obs_space = {
            'act': self.n_actions,
            'ack': len(ACK_STATE)
        }
        self._observation_space = [ obs_space[key] for key in OBS_ORDER ]

    @property
    def observation_space(self):
        return copy.deepcopy(self._observation_space)

    def reset(self):
        print("[Info] Resetting environment and initializing nodes...")
        self._t = 0
        self._agents_num = 1
        
        # Create normal nodes
        self._normal_nodes: list[Node] = []
        for _i, node_config in enumerate(self._scene_config['nodes']):
            node_id = _i+self._agents_num
            node = Node(node_id, self._env, node_config)
            if node_config['type'] == 'TDMA':
                mac_protocol = TDMA(node_config['strategy'])
            elif node_config['type'] == 'ALOHA':
                mac_protocol = ALOHA(node_config['strategy'])
            else:
                ValueError(f"Unsupported node type: {node['type']}")
            node.attach_protocol(mac_protocol)
            node.reset()
            self._normal_nodes.append(node)
            print('[Info] Create normal node id = {}, type = {}, delay = {}.'.format(node_id, node_config['type'], node.delay))
        
        # Create AI nodes
        node_config = self._scene_config['agents'][0]
        auv_model = AUVMobilityRMP(AP_COORD, self._space, self._targets['goals'], self._targets['padding'], speed=node_config['speed'])
        self._agent_node = Node(0, self._env, node_config, mobility_model=auv_model)
        mac_protocol = DRLMAC()
        self._agent_node.attach_protocol(mac_protocol)
        self._agent_node.reset()
        print('[Info] Create AI node id = 0, type = {}, delay = {}, speed = {}.'.format(
                self._agent_node.type, self._agent_node.delay, self._agent_node.mobility_model.speed))

        # Create AP node
        self.nodes_num = len(self._normal_nodes) + self._agents_num
        self._AP = AccessPoint(self.nodes_num)
        self._downlink_packets = {'idx': 0, 'max': 200, 'log': []}  # AP downlink packets

        # Set initial state
        init_obs = [0]*self.n_features
        self._actions_log = [0] # Log agent actions for each slot

        return init_obs

    def trigger_measure_delay(self):
        mac_protocol = self._agent_node.mac_protocol  # type: DRLMAC
        mac_protocol.measure_delay()

    def step(self, action):
        '''
        Execute the given agent action in the environment.

        Args:
            action (int): The action taken by the agent.

        Returns:
            - obs (list[int]): Observation results.
            - reward (int): Reward from the environment.
            - info (dict): Environment information, including ACK results and other agent-specific metadata.
        '''
        uplink_packets = []
        self._actions_log.append(action)
        
        # Part 1: Handle MAC protocol and slot allocation
        # =========================================================
        # Agent
        mac_protocol = self._agent_node.mac_protocol  # type: DRLMAC
        mac_protocol.set_action(action) # Inject action
        access_decision, send_packet = self._agent_node.mac_decide()
        if send_packet: uplink_packets.append(send_packet)
        
        # Normal nodes
        self._alloc_tdma_slots()        # Handle slot allocation requests for TDMA nodes
        for node in self._normal_nodes:
            access_decision, send_packet = node.mac_decide()
            if send_packet: uplink_packets.append(send_packet)

        # Part 2: Step simulation and update node states
        # =========================================================
        # Agent
        mac_info = self._agent_node.step(self._downlink_packets)

        # Normal nodes
        for node in self._normal_nodes:
            node.step(self._downlink_packets)

        # Access Point
        self._downlink_packets = self._AP.step(uplink_packets)

        # Part 3: Calculate reward and return observations
        # =========================================================
        self._t += 1

        comm_ack = mac_info['ack_frame']['state']
        measured_delay = mac_info['measured_delay']
        info = {
            'action': action,
            'ack': comm_ack,
            'is_moving': self._agent_node.mobility_model.get_speed() > 0,
            'actual_delay': self._agent_node.delay,
            'measured_delay': measured_delay
        }
        obs = [action, comm_ack]
        reward = self._reward(comm_ack)
        return obs, reward, info
    
    def _alloc_tdma_slots(self):
        mac_type = 'TDMA'
        alloc_request = [None] * self.nodes_num         # Store slot allocation requests for TDMA nodes
        for i, node in enumerate(self._normal_nodes):   # Process slot allocation requests
            if node.type == mac_type:
                mac_protocol = node.mac_protocol # type: TDMA
                strategy_request = mac_protocol.get_strategy_request()
                if strategy_request is not None:
                    alloc_request[node.id] = strategy_request
        if not all(x is None for x in alloc_request):
            nodes = []
            for i, node in enumerate(self._normal_nodes):
                if node.type == mac_type:
                    request_info = alloc_request[node.id]
                    nodes.append({
                            'id': node.id,
                            'strategy': node.mac_protocol._sent_slot if request_info is None else request_info,
                            'delay': node.delay
                        })
            alloc_res = allocate_tdma_slots(nodes)      # Perform slot allocation and update sending strategy for TDMA nodes
            for node in alloc_res:
                if alloc_request[node['id']] != None:
                    idx = node['id'] - self._agents_num
                    mac_protocol = self._normal_nodes[idx].mac_protocol # type: TDMA
                    mac_protocol.set_strategy(node['strategy'])

    def _reward(self, ack_result):
        '''
        Reward function.

        Args:
            ack_result (int): ACK reception result, which can be:
                - ACK_STATE['NO_ACK'] (0): No ACK received (indicates either no data frames were received or a collision occurred).
                - ACK_STATE['ACK'] (1): ACK received (indicates the data frame was successfully received).
                - ACK_STATE['OTHER_ACK'] (2): ACK received, but intended for a different node.

        Returns:
            int: Reward value.
                - 1: Data frame successfully received.
                - 0: No data frame received or collision occurred.
        '''

        if ack_result == ACK_STATE['ACK'] or ack_result == ACK_STATE['OTHER_ACK']:
            com_rew = 1
        else:
            com_rew = 0
        
        return com_rew