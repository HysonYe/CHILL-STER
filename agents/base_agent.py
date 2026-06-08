from abc import ABC, abstractmethod

class BaseAgent(ABC):
    '''
    Base agent class that defines basic interfaces and common methods.

    All specific agent implementations should inherit from this class and implement its abstract methods.

    The primary goal is to ensure that the training workflow in main.py remains compatible with different agent implementations.
    '''
    def __init__(self, gamma):
        self._gamma = gamma
        self.delay_aware = False
        self.t = 0

    def next_step(self):
        self.t += 1

    @property
    def request_measure_delay(self):
        return False

    # Abstract methods: concrete subclasses must implement these
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update_delay(self, observation = None, delay_info = {}):
        pass

    @abstractmethod
    def build_state(self, observation = None):
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, prob, observation):
        pass

    @abstractmethod
    def choose_action(self, states):
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        pass

    @abstractmethod
    def learn(self):
        pass
