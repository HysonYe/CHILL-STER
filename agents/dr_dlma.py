from agents.base_agent import BaseAgent
from agents.common.brain import DQN
from agents.common.memory import ReplayMemory
from infra.environment import OBS_ORDER
import copy
import torch

class DRDLMA(BaseAgent):
    '''
    DR-DLMA mac protocol
    '''
    def __init__(self, obs_dims, obs_len, obs_space, n_actions,
                memory_size = 500, batch_size = 32, replace_target_iter = 200, learning_rate = 0.01,
                gamma = 0.9, epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.995, steady_u = 0.05, win_size = 2000, oracle = False,
                device = 'cuda'):
        super().__init__(gamma)
        # copy params
        self._device = device.lower()
        self._steady_u = steady_u
        self._windows_size = win_size
        self._oracle = oracle   # If True, continuous training with accurate delay feedback (zero delay)
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._gamma = gamma
        self._epsilon = epsilon
        self._n_actions = n_actions
        self._obs_dims, self._obs_len, self._obs_space = obs_dims, obs_len, obs_space

        # Calculate state and action space dimensions
        self._act_idx = OBS_ORDER.index('act')
        self._ack_idx = OBS_ORDER.index('ack')
        self._one_hot_obs_dims = sum(obs_space)
        self._state_size = self._one_hot_obs_dims*self._obs_len
        self._brain = DQN(self._state_size, self._n_actions, 1, replace_target_iter, learning_rate,
                          gamma, epsilon, epsilon_min, epsilon_decay, device)
        self.type = 'DR-DLMA'
        self.reset()
    
    def reset(self):
        self._memory = JudiciousReplayMemory(self._memory_size, self._state_size, self._device)
        self._request_delay = True  # Flag to indicate if updated delay information is required
        self._delay = -1
        self._state = [0.] * self._state_size
        self._need_training = True
        self._record_action = []
        self._record_reward = []
        self._record_U = []
        self._record_s_U = 0.
        self.delay_aware = True
        self.t = 0

    def update_delay(self, observation = None, delay_info:dict = {}):
        if self._oracle:
            ref_delay = delay_info.get('actual_delay', -1)
            mac_name = self.type+'-Oracle'
        else:
            ref_delay = delay_info.get('measured_delay', -1)
            mac_name = self.type

        if self._delay != ref_delay:
            self._delay = ref_delay
            print(f'[Info] Agent {mac_name} receives beacon at steps {self.t}, updates delay information to {ref_delay}.')
        
    def build_state(self, observation = None):
        if observation is None: return copy.deepcopy(self._state)
        action = observation[self._act_idx]
        ack = observation[self._ack_idx]
        self._record_action.append(action)

        # Construct observation z = [a(t-2D), ack(t)]
        t, two_D = len(self._record_action)-1, 2*max(self._delay, 0)
        t_sub_2D = max(t-two_D, 0)
        z = [self._record_action[t_sub_2D], ack]

        # Convert observation z to one-hot encoding and update the state vector
        one_hot = []
        for idx, dim in zip(z, self._obs_space):
            v = [0] * dim
            v[idx] = 1
            one_hot.extend(v)
        self._state = self._state[self._one_hot_obs_dims:]+one_hot
        next_state = copy.deepcopy(self._state)
        return next_state
        
    def store_transition(self, state, action, reward, next_state, prob, observation):
        self._record_reward.append(reward)
        self._memory.push(state, action, reward, next_state, self._delay)

    def choose_action(self, states):
        probs = [0] # NOTE: Placeholder implementation

        s = torch.tensor(states, dtype=torch.float32, device=self._device)
        acts, sa_v = self._brain.choose_action(s)
        q_values = sa_v[torch.arange(sa_v.size(0)), acts]
        return acts, probs, q_values

    def save_checkpoint(self, path):
        self._brain.save(path)

    def learn(self):
        need_train, change_flag = self._is_need_training()
        if need_train:
            states, actions, rewards, next_states = self._memory.sample(self._batch_size)
            loss_info = self._brain.learn(states, actions, rewards, next_states)
            if change_flag:
                print(f'[Info] Agent DR-DLMA stops training at episode {self.t}.')
            return loss_info
        elif change_flag:
            print(f'[Info] Agent DR-DLMA starts training at episode {self.t}.')
            self._request_delay = True
        return None

    def request_measure_delay(self):
        ret = self._request_delay
        self._request_delay = False
        return ret

    def _is_need_training(self):
        '''
        DR-DLMA Component: Nimble Training Mechanism

        Returns:
            is_need (bool): Whether training is currently required.
            change_flag (bool): Whether the training status is about to change.
        '''
        if self._oracle: return True, False # Continuous training in oracle mode
        rew_len = min(self._windows_size, len(self._record_reward))
        U = sum(self._record_reward[-rew_len:]) / rew_len
        self._record_U.append(U)

        is_need = self._need_training   # Return previous flag
        if is_need:
            if len(self._record_U) >= self._windows_size:
                U_t = self._record_U[-1]
                U_past_N = self._record_U[-self._windows_size]
                if abs(U_t - U_past_N) / max(abs(U_past_N), 1e-5) < self._steady_u:
                    self._need_training = False
                    self._record_s_U = U_t  # Record current U as the stable value
        else:
            U_t = self._record_U[-1]
            U_s = self._record_s_U
            if abs(U_t - U_s) / max(abs(U_s), 1e-5) >= self._steady_u:
                self._need_training = True
        change_flag = is_need != self._need_training
        return is_need, change_flag

class JudiciousReplayMemory(ReplayMemory):
    def __init__(self, memory_size, state_size, device):
        super().__init__(memory_size, state_size, device)

    def reset(self):
        super().reset()
        self._memory_delay = torch.zeros((self._memory_size,), dtype=torch.int, device=self._device)
        self._delay_max = -1

    def push(self, state, action, reward, next_state, delay):
        self._update_delay_boundary(delay)
        idx = self._memory_counter % self._memory_size
        self._memory_delay[idx] = delay
        super().push(state, action, reward, next_state)

    def sample(self, batch_size):
        '''
        DR-DLMA Component: Judicious Experience Replay
        '''
        # Generate random indices
        current_count = min(self._memory_counter, self._memory_size)
        valid_count = current_count - self._delay_max * 2
        first_idx = self._memory_counter % self._memory_size if self._memory_counter >= self._memory_size else 0
        random_offsets = torch.randperm(valid_count, device=self._device)[:batch_size]
        idxs = (random_offsets + first_idx) % self._memory_size

        # Construct aligned transitions (St, At, Rt+2D+1, St+2D+1)
        samples = self._memory[idxs, :]
        samples_2D = self._memory[(idxs + 2 * self._memory_delay[idxs]) % self._memory_size, :]
        states = samples[:, :self._state_size]
        actions = samples[:, self._state_size].long()
        rewards = samples_2D[:, self._state_size + 1].unsqueeze(-1) # (batch_size, 1)
        next_states = samples_2D[:, -self._state_size:]

        return states, actions, rewards, next_states
    
    def _update_delay_boundary(self, delay):
        if self._delay_max == -1 and delay >= 0:
            self._memory_delay[:self._memory_counter] = delay
        self._delay_max = max(self._delay_max, delay)