from collections import Counter
from agents.base_agent import BaseAgent
from agents.common.brain import DQN
from agents.common.memory import ReplayMemory
from infra.environment import ACK_STATE, NODE_ACT, OBS_ORDER
import copy
import torch

class DLMAC(BaseAgent):
    '''
    DL-MAC mac protocol
    '''
    def __init__(self, obs_dims, obs_len, obs_space, n_actions,
                memory_size = 500, batch_size = 32, replace_target_iter = 200, learning_rate = 0.01, decision_interval = 6,
                gamma = 0.9, epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.995, async_mode = False, enhance_mode = False,
                device = 'cuda'):
        super().__init__(gamma)
        # copy params
        self._device = device.lower()
        self._async_mode = async_mode
        self._enhance_mode = enhance_mode
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._decision_interval = decision_interval
        self._n_actions = n_actions
        self._gamma = gamma
        self._epsilon = epsilon
        self._obs_dims, self._obs_len, self._obs_space = obs_dims, obs_len, obs_space

        # Calculate state and action space dimensions
        self._act_idx = OBS_ORDER.index('act')
        self._ack_idx = OBS_ORDER.index('ack')
        if self._async_mode:
            # For async-DL-MAC, the action space includes picking one slot to transmit within
            # the current decision interval, or choosing not to transmit (WAIT).
            self._n_actions = decision_interval + 1
            self._obs_space[self._act_idx] = self._n_actions
        self._one_hot_obs_dims = sum(self._obs_space)
        if self._enhance_mode:
            # Enhanced mode: provides additional ACK information across the entire decision window.
            self._obs_space += [self._obs_space[self._ack_idx]] * (self._decision_interval-1)
            self._one_hot_obs_dims = sum(self._obs_space)
        self._state_size = self._one_hot_obs_dims*self._obs_len
        self._brain = DQN(self._state_size, self._n_actions, 1, replace_target_iter, learning_rate,
                          gamma, epsilon, epsilon_min, epsilon_decay, device)
        self.type = 'DL-MAC'
        self.reset()

    def reset(self):
        self._memory = ReplayMemory(self._memory_size, self._state_size, self._device)
        self._delay = -1
        self._tx_delay = 0                      # Transmission delay (slot offset) for async-DL-MAC within current decision window
        self._state = [0.] * self._state_size
        self._window_observations = []          # Record observations within each decision window
        self._cur_obs = [0,0]                   # Current slot observations (act and ack)
        self.t = 0

    def update_delay(self, observation = None, delay_info:dict = {}):
        '''
        Placeholder implementation

        In DL-MAC, this interface is not strictly required; it is kept here only 
        for compatibility with the training workflow in main.py.
        '''
        ref_delay = delay_info.get('actual_delay', -1)
        if ref_delay != self._delay:
            self._delay = ref_delay
            print(f'[Info] Agent DL-MAC at steps {self.t}, updates delay information to {ref_delay}.')

    def build_state(self, observation = None):
        if observation is None: return copy.deepcopy(self._state)
        self._window_observations.append(observation)
        next_t = self.t + 1
        if next_t % self._decision_interval == 0:
            # step 1. Determine at
            action = self._tx_delay                     # Use the previously determined transmission delay as the action info for the current window

            # step 2. Get ACK sequence to determine ct
            ack_seq = [obs[self._ack_idx] for obs in self._window_observations]
            self._window_observations = []

            # step 3. Construct the next state
            if self._enhance_mode:
                s_ = [action] + ack_seq                 # Enhanced DL-MAC includes the full ACK sequence within the decision window
            else:
                ack = self._ack_seq_to_scalar(ack_seq)  # ct: Original DL-MAC state contains only a scalar ACK value
                s_ = [action, ack]
            self._cur_obs = s_

            one_hot = []
            for idx, dim in zip(s_, self._obs_space):   # Convert state to one-hot encoding
                v = [0] * dim
                v[idx] = 1
                one_hot.extend(v)
            self._state = self._state[self._one_hot_obs_dims:]+one_hot
            next_state = copy.deepcopy(self._state)
            return next_state

        return copy.deepcopy(self._state)

    def store_transition(self, state, action, reward, next_state, prob, observation):
        next_t = self.t + 1
        if next_t % self._decision_interval == 0:
            really_reward = self._reward_func(self._cur_obs[self._ack_idx:])
            really_action = self._tx_delay
            self._memory.push(state, really_action, really_reward, next_state)

    def choose_action(self, states):
        probs = [0] # NOTE: Placeholder implementation

        # Decision making
        _eps = self._brain._eps
        s = torch.tensor(states, dtype=torch.float32, device=self._device)
        acts, sa_v = self._brain.choose_action(s)
        q_values = sa_v[torch.arange(sa_v.size(0)), acts]
        delay_slot = self.t % self._decision_interval
        if delay_slot == 0:
            self._tx_delay = acts[0].item()
        else:
            self._brain._eps = _eps             # Keep epsilon constant at non-decision steps to prevent premature decay

        # Action generation
        acts.fill_(NODE_ACT['WAIT'])
        if self._tx_delay == delay_slot:
            if self._async_mode or delay_slot == 0:
                '''
                Async mode: execute SEND whenever the current slot matches the selected transmission delay.
                Sync mode: execute SEND only at the first slot of the decision window.
                '''
                acts.fill_(NODE_ACT['SEND'])

        return acts, probs, q_values
    
    def save_checkpoint(self, path):
        self._brain.save(path)

    def learn(self):
        states, actions, rewards, next_states = self._memory.sample(self._batch_size)
        loss_info = self._brain.learn(states, actions, rewards, next_states)
        return loss_info

    def _reward_func(self, ack_raw: list):
        '''
        Calculates an accurate reward since the algorithm's decision interval differs from the environment's time steps.

        The logic is consistent with the reward function in environment.py: a reward is granted only 
        when the feedback state is ACK or OTHER_ACK.

        Args:
            ack_raw: Raw ACK sequence within the decision window, with length equal to the decision interval.
        
        Returns:
            com_rew: The calculated reward value.
        '''
        if self._enhance_mode:
            ack_scalar = self._ack_seq_to_scalar(ack_raw)
            if ack_scalar == ACK_STATE['ACK'] or ack_scalar == ACK_STATE['OTHER_ACK']:
                com_rew = 1
            else:
                com_rew = 0
                # In enhanced mode, rewards are provided proportionally for OTHER_ACK occurrences within 
                # the decision window to encourage transmission without causing collisions.
                com_rew += ack_raw.count(ACK_STATE['OTHER_ACK']) / len(ack_raw)
        else:
            ack_scalar = ack_raw[0] # This value has been converted to scalar ct in build_state
            if ack_scalar == ACK_STATE['ACK'] or ack_scalar == ACK_STATE['OTHER_ACK']:
                com_rew = 1
            else:
                com_rew = 0
        
        return com_rew
    
    def _ack_seq_to_scalar(self, ack_seq):
        '''
        Converts an ACK sequence within a decision window into a scalar ACK, 
        replicating the definition of the scalar ct in DL-MAC.

        Args:
            ack_seq: ACK sequence within the decision window.
        
        Returns:
            ack: The resulting scalar ACK value.
        '''
        ack_counter = Counter(ack_seq)
        seq_len = len(ack_seq)
        if ack_counter[ACK_STATE['ACK']] > 0:
            ack = ACK_STATE['ACK']
        elif ack_counter[ACK_STATE['OTHER_ACK']] == seq_len:
            ack = ACK_STATE['OTHER_ACK']
        elif ack_counter[ACK_STATE['NO_ACK']] == seq_len:
            ack = ACK_STATE['NO_ACK']
        else:
            ack = ACK_STATE['NO_ACK']
            
        return ack