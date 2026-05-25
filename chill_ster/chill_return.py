import torch
from torch.distributions.categorical import Categorical

def get_epsilon_greedy_dist(n_actions:int, epsilons: torch.Tensor, optimal_acts: torch.Tensor, device) -> Categorical:
    '''
    Calculates the action distribution for an epsilon-greedy policy.

    Args:
        n_actions (int): Number of available actions.
        epsilons (torch.Tensor): Exploration rates, shape (batch_size,).
        optimal_acts (torch.Tensor): Indices of optimal actions, shape (batch_size,).
        device (str): Device descriptor (e.g., 'cpu' or 'cuda').

    Returns:
        dist (torch.distributions.Categorical): The resulting action distribution.
    '''
    batch_size = optimal_acts.shape[0]
    avg_prob = epsilons / n_actions
    probabilities = torch.zeros((batch_size, n_actions), dtype=torch.float32, device=device)
    probabilities += avg_prob.unsqueeze(1)
    arange_batch = torch.arange(batch_size, device=device)
    probabilities[arange_batch, optimal_acts] += (1.0 - epsilons)
    dist = Categorical(probs=probabilities)
    return dist

class CHILLReturn:
    '''
    Credit Horizon-Limited λ-Return (CHILL-Return)

    This class implements a delay-robust credit assignment mechanism for underwater
    acoustic networks by combining a fixed credit horizon with λ-return bootstrapping.
    It is designed to learn from asynchronously observed rewards without requiring
    real-time ranging, while still preserving the reward signal within a horizon that
    covers the maximum propagation delay.

    By truncating the return horizon and applying clipped importance sampling, CHILL-Return
    reduces the variance and off-policy bias caused by multi-step returns. The result is
    a stable target for deep reinforcement learning under mobile and time-varying underwater
    communication conditions.
    '''
    def __init__(self, actions: int, horizon: int, gamma: float, lamb: float, beta: float = 0.2):
        '''
        Initializes the CHILL-Return mechanism with the specified parameters.

        Args:
            actions (int): Size of the action space (used for epsilon-greedy distribution).
            horizon (int): Finite credit horizon length (Horizon H), should satisfy H >= 2 * D_max + 1.
            gamma (float): Discount factor.
            lamb (float): Lambda parameter for balancing bias and variance.
            beta (float): Exponent for importance sampling weight correction.
        '''
        self._n_actions = actions
        self._n_step = horizon
        self._gamma = gamma
        self._gl = gamma * lamb
        self._gl_pow_n = self._gl**self._n_step
        self._beta = beta

    def _compute_traj_probs(self, probs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Computes the sum of log-probabilities for n-step trajectories.

        Args:
            probs (torch.Tensor): Log-probabilities of actions, shape (batch_size,).
            masks (torch.Tensor): Trajectory continuity masks, shape (batch_size,), where 1 indicates continuity and 0 indicates end of trajectory.
        Returns:
            res (torch.Tensor): Sum of n-step trajectory log-probabilities, shape (batch_size,).
        '''
        batch_size = probs.size(0)
        res = torch.zeros_like(probs)   # Initialize results
        '''
        NOTE: When evaluating the action value Q(s,a), the initial action is a given condition,
        so its log-likelihood contribution is 0. This is appropriate for value-based algorithms
        (e.g., DQN/DDPG/SAC) as they directly evaluate the expected return of an action.
        However, if extending to policy-based algorithms (e.g., A2C/A3C/PPO/TRPO), the calculation
        might need adjustment to include the probability contribution of the initial action.
        '''
        active_mask = masks.clone()     # active_mask[i] indicates whether the trajectory starting from i to i+k is continuous
        for k in range(1, self._n_step):# Iterate to look ahead k steps
            if k >= batch_size:
                break
            valid_len = batch_size - k
            shifted_probs = probs[k:]
            res[:valid_len] += shifted_probs * active_mask[:valid_len]
            if k < self._n_step - 1:
                active_mask[:valid_len] *= masks[k:]
                active_mask[valid_len:] = 0
        return res

    def _compute_weights(self, old_probs: torch.Tensor, new_probs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Computes the importance sampling weights in experience replay.

        Args:
            old_probs (torch.Tensor): Log-probabilities of actions under the behavior policy, shape (batch_size,).
            new_probs (torch.Tensor): Log-probabilities of actions under the current policy, shape (batch_size,).
            masks (torch.Tensor): Trajectory continuity masks, shape (batch_size,).
        
        Returns:
            weights (torch.Tensor): Importance sampling weights, shape (batch_size, 1), where w_i = min( ρ^β, 1 ).
        '''
        old_traj_probs = self._compute_traj_probs(old_probs, masks)
        new_traj_probs = self._compute_traj_probs(new_probs, masks)
        default_w = (new_traj_probs - old_traj_probs).exp()
        weights = torch.pow(default_w, self._beta).clip(0, 1).reshape((-1, 1))
        return weights

    def _compute_H_step_return_gpu(self, sv_td_errs: torch.Tensor, masks: torch.Tensor, state_value: torch.Tensor):
        '''
        Efficiently computes the H-step return on GPU using unfolding, with O(NM) complexity.
        '''
        T, objs = sv_td_errs.shape
        device = sv_td_errs.device
        # Construct the convolution kernel
        kernel = (self._gl ** torch.arange(self._n_step, device=device))

        # Create a sliding window view of td_errs
        td_zeros = torch.zeros(self._n_step - 1, objs, device=device)
        td_padded = torch.cat([sv_td_errs, td_zeros], dim=0)
        td_windows = td_padded.unfold(0, self._n_step, 1)  # (T, objs, n_step)

        # Create a sliding window view of masks
        m_zeros = torch.zeros(self._n_step - 1, device=device)
        m_padded = torch.cat([masks, m_zeros], dim=0)
        m_windows = m_padded.unfold(0, self._n_step, 1)   # (T, n_step)

        # Calculate mask product, stopping cumulative multiplication when a mask is 0
        m_mask_base = torch.cat([torch.ones(T, 1, device=device), m_windows[:, :-1]], dim=1)
        m_mask = m_mask_base.cumprod(dim=1).unsqueeze(1)    # (T, 1, n_step)

        # Calculate GAE within the window
        windowed_gae = (td_windows * m_mask * kernel).sum(dim=-1)
        
        h_step_returns = state_value + windowed_gae
        return h_step_returns

    def _compute_H_step_return_cpu(self, sv_td_errs: torch.Tensor, masks: torch.Tensor, state_value: torch.Tensor):
        '''
        Implementation using a CPU loop with O(N) complexity.
        Native languages like C/C++ would be more efficient, capable of completion within 0.01ms.
        In Python, this approach is slower due to loop overhead.
        Consider using PyTorch's load_inline (C/C++ extension) for acceleration.
        '''
        batch_size, objs = sv_td_errs.shape
        device = sv_td_errs.device
        h_step_returns = torch.zeros((batch_size, objs), device=device)
        cur_gae = torch.tensor(0.0, device=device, dtype=sv_td_errs.dtype)
        traj_len = 0                                    # Length of the current continuous trajectory
        for t in reversed(range(batch_size)):
            if masks[t] == 0:                           # Reset GAE and trajectory length upon end-of-trajectory flag
                traj_len = 0
                cur_gae.fill_(0.)
            cur_gae = sv_td_errs[t] + self._gl * cur_gae
            if traj_len >= self._n_step:
                cur_gae -= self._gl_pow_n * sv_td_errs[t + self._n_step]
            h_step_returns[t] = state_value[t] + cur_gae
            traj_len += 1

        return h_step_returns
    
    def _compute_H_step_return(self, rewards: torch.Tensor, tar_q_eval: torch.Tensor,
                               tar_next_q_eval: torch.Tensor, masks: torch.Tensor):
        max_q, _ = torch.max(tar_q_eval, dim=1)
        next_max_q, _ = torch.max(tar_next_q_eval, dim=1)
        sv_td_errs = rewards + self._gamma * next_max_q - max_q    # State-Value TD Error
        '''
        NOTE: Here we compute a state-value TD error. Because DQN uses an epsilon-greedy
        policy, the state value V(S_t) equals max_a Q(S_t, a). This is why we use max_q
        directly when computing the state-value TD error instead of q_eval. If extending
        this implementation to other algorithms, the method for computing the state value
        should be adapted accordingly.
        '''
        if tar_q_eval.is_cuda:
            target_q = self._compute_H_step_return_gpu(sv_td_errs.detach(), masks, max_q.detach())
        else:
            target_q = self._compute_H_step_return_cpu(sv_td_errs.detach(), masks, max_q.detach())
        return target_q

    def forward_loss(self, 
                     q_eval: torch.Tensor, 
                     actions: torch.Tensor, 
                     rewards: torch.Tensor, 
                     target_q_eval: torch.Tensor,
                     target_next_q_eval: torch.Tensor,
                     old_probs: torch.Tensor, 
                     new_probs: torch.Tensor, 
                     masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CHILL-Return, calculating the loss with importance sampling weights and finite-horizon returns.
        
        Args:
            q_eval (torch.Tensor): Q-values for all actions from the evaluation network, shape (batch_size, n_actions).
            actions (torch.Tensor): Actual actions sampled from experience replay, shape (batch_size,).
            rewards (torch.Tensor): Reward sequence, shape (batch_size, 1).
            target_q_eval (torch.Tensor): Q-values for all actions from the target network at t, shape (batch_size, n_actions).
            target_next_q_eval (torch.Tensor): Q-values for all actions from the target network at t+1, shape (batch_size, n_actions).
            old_probs (torch.Tensor): Log-probabilities of actions under the behavior policy, shape (batch_size,).
            new_probs (torch.Tensor): Log-probabilities of actions under the current policy, shape (batch_size,).
            masks (torch.Tensor): Trajectory continuity masks, shape (batch_size,).
            
        Returns:
            loss (torch.Tensor): Weighted mean squared loss incorporating importance sampling and finite-horizon returns (scalar).
        """
        batch_size = q_eval.size(0)
        batch_index = torch.arange(batch_size, device=q_eval.device, dtype=torch.long)

        # Compute H-step return target
        target_q = self._compute_H_step_return(rewards, target_q_eval, target_next_q_eval, masks)

        # Compute importance sampling weights
        weights = self._compute_weights(old_probs, new_probs, masks)

        # Calculate final weighted squared loss
        current_q = q_eval[batch_index, actions].view(-1, 1)
        td_errors = current_q - target_q.to(current_q.dtype)
        loss = (weights * td_errors.pow(2)).sum() / batch_size
        return loss