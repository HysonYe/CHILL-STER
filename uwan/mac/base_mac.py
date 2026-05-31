from abc import ABC, abstractmethod
from uwan.nodes import Node

class BaseMAC(ABC):
    def __init__(self, protocol_type: str):
        self.protocol_type = protocol_type
        self._node = None

    def attach_to_node(self, node: Node):
        self._node = node

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def decide(self):
        pass

    @abstractmethod
    def step(self, downlink_packets=None):
        pass