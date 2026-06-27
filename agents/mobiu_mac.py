from agents.base_agent import BaseAgent
from agents.common.brain import DQN
from infra.environment import OBS_ORDER, NODE_ACT, ACK_STATE
from infra.utils.tools import up_to_integer
from chill_ster import CHILLReturn, STER, get_epsilon_greedy_dist
from collections import deque
import torch, copy
import numpy as np

class MobiUMAC(BaseAgent):
    '''
    MobiU-MAC protocol
    '''
    def __init__(self, obs_dims, obs_len, obs_space, n_actions, horizon, lamb, beta,
                isa_est_wind, sub_eff_len, ster_radius, ster_decay = 0.95, memory_size = 500, batch_size = 32,
                replace_target_iter = 200, learning_rate = 1e-3, gamma = 0.9, epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.995,
                device = 'cuda'):
        super().__init__(gamma)
        # copy params
        self._device = device.lower()
        self._gamma = gamma
        self._epsilon = epsilon
        self._n_actions = n_actions
        self._obs_dims, self._obs_len, self._obs_space = obs_dims, obs_len, obs_space
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._isa_est_wind = isa_est_wind
        self._ster_decay = ster_decay
        self._ster_radius = ster_radius
        self._n_sub_traj_effective_len = sub_eff_len
        self._horizon = horizon
        self._lambda = lamb
        self._beta = beta

        # Calculate state and action space dimensions
        self._ack_idx = OBS_ORDER.index('ack')
        self._one_hot_obs_dims = sum(obs_space)
        self._state_size = self._one_hot_obs_dims*self._obs_len
        self._brain = CHILLDQN(self._state_size, self._n_actions, 1, replace_target_iter, learning_rate,
                               gamma, epsilon, epsilon_min, epsilon_decay, horizon, lamb, beta, device=device)
        self.type = 'MobiU-MAC'
        self.reset()

    def reset(self):
        rew_size = 1
        self._eps_buffer = {}
        self._default_anchor = -1
        self._isa =  self._default_anchor
        self._memory = STER(self._memory_size, self._state_size, rew_size, self._default_anchor,
                            self._ster_radius, self._ster_decay, self._device)
        self._state = [0.] * self._state_size
        self._isa_estimator = ImplicitAnchorEstimator(self._horizon, self._isa_est_wind, NODE_ACT, ACK_STATE, OBS_ORDER)
        self.t = 0

    def update_delay(self, observation = None, ref_delay = -1):
        isa = self._isa
        
        if self._isa_est_wind > 0 and observation is not None:
            est_isa = self._isa_estimator.push(observation)
            if est_isa is not None:
                if abs(self._isa - est_isa) <= 1 or self._isa == -1:
                    isa = est_isa
        elif self._isa_est_wind == 0:
            isa = ref_delay

        if isa not in self._eps_buffer:
            self._eps_buffer[isa] = self._epsilon

        if isa != self._isa:
            print(f"[Info] Time step {self.t}: ISA updated from {self._isa} to {isa}.")
            pre_isa = self._isa
            self._isa = isa
            if pre_isa != self._default_anchor:
                self._eps_buffer[pre_isa] = self._brain._eps
                self._brain._eps = self._eps_buffer[isa]

    def build_state(self, observation = None):
        if observation is None: return copy.deepcopy(self._state)

        # Convert state to one-hot encoding
        one_hot = []
        for idx, dim in zip(observation, self._obs_space):
            v = [0] * dim
            v[idx] = 1
            one_hot.extend(v)
        self._state = self._state[self._one_hot_obs_dims:]+one_hot
        next_state = copy.deepcopy(self._state)
        return next_state
    
    def store_transition(self, state, action, reward, next_state, prob):
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        }
        self._memory.push(transition, prob, self._isa)

    def choose_action(self, states):
        s = torch.tensor(states, dtype=torch.float32, device=self._device)
        acts, log_probs, sa_v = self._brain.choose_action(s)
        q_values = sa_v[torch.arange(sa_v.size(0)), acts]
        return acts, log_probs, q_values

    def save_checkpoint(self, path):
        self._brain.save(path)

    def learn(self):
        pass

    def _sample(self):
        n_sampling_len = self._n_sub_traj_effective_len + self._horizon - 1
        useful_len = self._n_sub_traj_effective_len
        batch_size = self._batch_size // useful_len
        trajs_num, anchors, transitions, probs, masks = self._memory.sample(batch_size, n_sampling_len)
        states, actions, rewards, next_states = transitions['states'], transitions['actions'],\
                                                transitions['rewards'], transitions['next_states']
        
        # Compute new_probs for the sampled transitions
        self._eps_buffer[self._isa] = self._brain._eps
        if anchors[0] == self._default_anchor:
            eps_array = np.full_like(anchors, self._eps_buffer[self._default_anchor], dtype=float)
        else:
            max_key = max(self._eps_buffer.keys())
            lookup_table = np.zeros(max_key + 1)
            for key, value in self._eps_buffer.items(): lookup_table[key] = value
            eps_array = lookup_table[anchors]
        new_probs = self._brain.compute_current_probs(states, actions, eps_array)

        # Compute indices for sub-trajectory sampling
        offsets = torch.arange(trajs_num, device=self._device).view(-1, 1) * n_sampling_len          
        local_indices = torch.arange(useful_len, device=self._device).view(1, -1)                    
        indices = (offsets + local_indices).view(-1)
        return states, actions, rewards, next_states, probs, new_probs, masks, indices
    
class CHILLDQN(DQN):
    def __init__(
        self, state_dims, n_actions, n_nodes,
        replace_target_iter=200, learning_rate=1e-3, gamma=0.9,
        epsilon=1., epsilon_min=0.01, epsilon_decay=0.995,
        horizon=10, lamb=0.95, beta=0.2,  # CHILL-Return specific parameters
        device='cuda'
    ):
        super().__init__(
            state_dims, n_actions, n_nodes,
            replace_target_iter, learning_rate, gamma,
            epsilon, epsilon_min, epsilon_decay, device
        )
        
        self._chill_return = CHILLReturn(
            actions=self._n_actions,
            horizon=horizon,
            gamma=self._gamma,
            lamb=lamb,
            beta=beta
        )
        print(f"[Info] CHILLDQN initialized successfully. Horizon H={horizon}, Lambda={lamb}, Beta={beta}")

    def choose_action(self, state):
        eps = self._eps
        acts, sa_v = super().choose_action(state)

        # NOTE: compute behavior policy log probabilities (old_probs) for the chosen actions
        _, optimal_act = torch.max(sa_v, dim=1)
        epsilons = torch.full((self._n_nodes,), eps, device=self._net_device)
        dist = get_epsilon_greedy_dist(self._n_actions, epsilons, optimal_act, self._net_device)
        log_probs = dist.log_prob(acts)
        
        return acts, log_probs, sa_v
    
    def compute_current_probs(self, states, actions, epsilons):
        # NOTE: compute current policy log probabilities (new_probs) for the given actions
        self._model.eval()
        with torch.amp.autocast(device_type=self._device_name, dtype=self._amp_dtype):
            with torch.no_grad():
                sa_v = self._model(states)
                _, optimal_act = torch.max(sa_v, dim=1)
                dist = get_epsilon_greedy_dist(self._n_actions, epsilons, optimal_act, self._net_device)
                log_probs = dist.log_prob(actions)
                return log_probs

    def learn(self, states, actions, rewards, next_states, old_probs, new_probs, masks, indices=None):
        self._model.train()
        self._learn_step += 1
        states_train, actions_train = states, actions
        if indices is not None: states_train, actions_train = states[indices], actions[indices]

        with torch.amp.autocast(device_type=self._device_name, dtype=self._amp_dtype):
            q_eval_train = self._model(states_train)
            with torch.no_grad():
                batch_size_full = states.size(0)
                combined_states = torch.cat([states, next_states], dim=0)
                combined_target_q = self._target_model(combined_states)
                target_q_eval = combined_target_q[:batch_size_full]
                target_next_q_eval = combined_target_q[batch_size_full:]

            q_loss = self._chill_return.forward_loss(
                q_eval=q_eval_train,
                actions=actions_train,
                rewards=rewards,
                target_q_eval=target_q_eval,
                target_next_q_eval=target_next_q_eval,
                old_probs=old_probs,
                new_probs=new_probs,
                masks=masks,
                indices=indices
            )

        self._optimizer.zero_grad(set_to_none=True)
        if self._device_name == 'cpu':
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
        else:
            self._amp_scaler.scale(q_loss).backward()
            self._amp_scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._amp_scaler.step(self._optimizer)
            self._amp_scaler.update()

        if self._learn_step % self._rt_iter == 0:
            self._replace_target_params()

        return {'RL_loss': q_loss.item()}

class ImplicitAnchorEstimator:

    def __init__(self, max_offset, window_size,
                 node_act=None, ack_state=None, obs_order:list =None):
        # copy params
        self._max_offset = max_offset
        self._window_size = window_size
        self._node_act  = node_act
        self._ack_state = ack_state
        self._act_idx = obs_order.index('act')
        self._ack_idx = obs_order.index('ack')

        # internal states
        self._steps = window_size - max_offset
        self._buf = deque(maxlen=window_size)
        self._offset_deque = deque(maxlen=20)
        self._offset_acc = 0
        self._scores = np.full(max_offset + 1, 0, dtype=np.int32)
        self._scores[:2] = -10**9
        self._init_flag = False

    def _compute_score(self, act, ack):
        if act == self._node_act['SEND']:
            if   ack == self._ack_state['ACK']:        return  1
            elif ack == self._ack_state['OTHER_ACK']:  return -1
            else: return 0
        elif act == self._node_act['WAIT']:
            if   ack == self._ack_state['NO_ACK']:     return  1
            elif ack == self._ack_state['ACK']:        return -1
            else: return 0
    
    def _smooth_offset(self, optimal_offset):
        self._offset_acc += optimal_offset
        if len(self._offset_deque) == self._offset_deque.maxlen:
            self._offset_acc -= self._offset_deque[0]
        self._offset_deque.append(optimal_offset)
        return self._offset_acc / len(self._offset_deque)

    def push(self, obs):
        '''
        Pushes a new observation and returns the estimated implicit spatial anchor z_t.
        Returns None if the sliding window is not yet full.
        '''
        buf, W, R = self._buf, self._window_size, self._max_offset + 1

        # step 1. Remove old observation and deduct the corresponding score
        if len(buf) == W:
            old_act = buf[0][self._act_idx]
            for offset in range(2, R):
                old_ack = buf[offset][self._ack_idx]
                self._scores[offset] -= self._compute_score(old_act, old_ack)

        # step 2. Add new observation and add the corresponding score
        buf.append(obs)
        if len(buf) < W: return None
        
        if self._init_flag == False:
            # Initialization
            self._init_flag = True
            for offset in range(2, R):
                for t in range(self._steps):
                    self._scores[offset] += self._compute_score(buf[t][self._act_idx], buf[t + offset][self._ack_idx])
        else:
            t = self._steps - 1
            act_in = buf[t][self._act_idx]
            for offset in range(2, R):
                ack_in = buf[t + offset][self._ack_idx]
                self._scores[offset] += self._compute_score(act_in, ack_in)

        # step 3. Select optimal offset and apply smoothing to get final implicit spatial anchor estimate
        best_offset = int(np.argmax(self._scores))
        smoothed_m = self._smooth_offset(best_offset)
        isa = up_to_integer(smoothed_m / 2)
        return isa