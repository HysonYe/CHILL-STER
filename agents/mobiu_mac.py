from agents.base_agent import BaseAgent
from agents.common.brain import DQN
from infra.environment import OBS_ORDER
from infra.utils.tools import up_to_integer
from chill_ster import CHILLReturn, STER, get_epsilon_greedy_dist
from collections import deque
import torch
import numpy as np

class MobiUMAC(BaseAgent):
    '''
    MobiU-MAC protocol
    '''
    def __init__(self, obs_dims, obs_len, obs_space, n_actions, n_step, lamb, beta, importance_sampling, clip,
                isa_est_wind, sub_eff_len, ster_radius, ster_decay = 0.95, memory_size = 500, batch_size = 32,
                replace_target_iter = 200, learning_rate = 1e-3, gamma = 0.9, epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.995,
                device = 'cuda'):
        super().__init__(gamma)
        # copy params
        self._device = device.lower()
        self._n_step = n_step
        self._lambda = lamb
        self._gamma = gamma
        self._epsilon = epsilon
        self._beta = beta
        self._clip = clip
        self._able_IS = importance_sampling
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._isa_est_wind = isa_est_wind
        self._ster_decay = ster_decay
        self._ster_radius = ster_radius
        self._n_sub_traj_effective_len = sub_eff_len    # 经验回放中每条轨迹的有效长度，过短会导致样本计算效率过低，过长会导致样本独立性过低
        self._n_actions = n_actions
        self._obs_dims, self._obs_len, self._obs_space = obs_dims, obs_len, obs_space

        # Calculate state and action space dimensions
        self._ack_idx = OBS_ORDER.index('ack')
        self._one_hot_obs_dims = sum(obs_space)
        self._state_size = self._one_hot_obs_dims*self._obs_len
        # self._brain = 
        self.type = 'MobiU-MAC'
        self.reset()

    def reset(self):
        self.t = 0

    def update_delay(self, observation = None, ref_delay = -1):
        pass

    def build_state(self, observation = None):
        pass
    
    def store_transition(self, state, action, reward, next_state, prob, observation):
        pass

    def choose_action(self, states):
        pass

    def save_checkpoint(self, path):
        pass

    def learn(self):
        pass

class CHILLDQN(DQN):
    def __init__(
        self, state_dims, n_actions, n_nodes,
        replace_target_iter=200, learning_rate=1e-3, gamma=0.9,
        epsilon=1., epsilon_min=0.01, epsilon_decay=0.995,
        horizon=10, lamb=0.95, beta=0.2,  # CHILL-Return 特有参数
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

    def learn(self, states, actions, rewards, next_states, old_probs, masks):
        self._model.train()
        self._learn_step += 1
        batch_size = states.size(0)
        with torch.amp.autocast(device_type=self._device_name, dtype=self._amp_dtype):
            q_eval = self._model(states)
            with torch.no_grad():
                combined_states = torch.cat([states, next_states], dim=0)
                combined_target_q = self._target_model(combined_states)
                target_q_eval = combined_target_q[:batch_size]
                target_next_q_eval = combined_target_q[batch_size:]
                _, optimal_act = torch.max(q_eval, dim=1)
                epsilons = torch.full((batch_size,), self._eps, device=self._net_device)
                dist = get_epsilon_greedy_dist(self._n_actions, epsilons, optimal_act, self._net_device)
                new_probs = dist.log_prob(actions)

            q_loss = self._chill_return.forward_loss(
                q_eval=q_eval,
                actions=actions,
                rewards=rewards,
                target_q_eval=target_q_eval,
                target_next_q_eval=target_next_q_eval,
                old_probs=old_probs,
                new_probs=new_probs,
                masks=masks
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