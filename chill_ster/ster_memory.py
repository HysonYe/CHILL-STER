import math
import torch
import numpy as np
from collections import deque
from infra.utils.tools import get_context_batchs, ripple_traverse

class STER:
    '''
    Spatio-Temporal Experience Replay (STER)

    This class implements a spatially-aware experience replay mechanism that dynamically
    adjusts sampling range and weights based on the current spatial context. It mitigates
    non-stationarity arising from motion in underwater acoustic networks by incorporating
    smoothed spatial context information to guide sample selection.
    '''

    def __init__(self, memory_size, state_size, reward_size, pad_spatial_anchor, radius=3, context_decay=0.95, device='cuda'):
        '''
        Initializes the Spatio-Temporal Experience Replay (STER) buffer.
        
        Args:
            memory_size (int): The maximum number of transitions to store for each spatial anchor.
            state_size (int): The dimensionality of the state representation.
            reward_size (int): The dimensionality of the reward representation.
            pad_spatial_anchor (int): The spatial anchor used for unknown spatial anchors.
            radius (int): The spatial radius for sampling neighboring anchors, default is 3.
            context_decay (float): The decay factor for updating spatial context, default is 0.95.
            device (str): The device to store the buffers, default is 'cuda'.
        '''
        # memory parameters
        self._device = device.lower()
        self._memory_size = memory_size
        self._state_size = state_size
        self._reward_size = reward_size

        # STER parameters
        self._pad_isa = pad_spatial_anchor
        self._isa = self._pad_isa
        self._spatial_context = self._pad_isa
        self._radius = radius
        self._context_decay = context_decay

        # buffers
        self._visited_anchors = set()
        self._anchor_buffers = {}

    def _init_anchor_buffer(self, impl_spatial_anchor):
        trajs = torch.zeros((self._memory_size, self._state_size*2+1+self._reward_size),
                             device=self._device)                       # Store multi-step transitions to form trajectories
        probs = torch.zeros((self._memory_size), device=self._device)   # Store sampling probabilities for each action
        ends = torch.ones((self._memory_size), device=self._device)     # Used to determine the continuity of the trajectory
        counter = { 
                    'total': 0,                                         # Total number of samplable transitions in current buffer
                    'ids': deque(maxlen=self._memory_size)              # Indices of new trajectory starts for fast localization
                    }
                    
        self._anchor_buffers[impl_spatial_anchor] = {
            'trajs': trajs,
            'probs': probs,
            'ends': ends,
            'counter': counter
        }

        self._visited_anchors.add(impl_spatial_anchor)

    def push(self, transition, prob, impl_spatial_anchor):
        if impl_spatial_anchor not in self._visited_anchors:
            self._init_anchor_buffer(impl_spatial_anchor)
            if self._isa == self._pad_isa and impl_spatial_anchor != self._pad_isa:
                # First time obtaining spatial anchor, copy the experience data from the pad anchor to the new anchor
                self._anchor_buffers[impl_spatial_anchor] = self._anchor_buffers[self._pad_isa]

        trajs_buffer = self._anchor_buffers[impl_spatial_anchor]['trajs']
        probs_buffer = self._anchor_buffers[impl_spatial_anchor]['probs']
        ends_buffer = self._anchor_buffers[impl_spatial_anchor]['ends']
        counter = self._anchor_buffers[impl_spatial_anchor]['counter']
        idx = counter['total'] % self._memory_size

        # Handle trajectory boundaries
        is_new_trajectory = False
        if counter['total'] == 0:
            counter['ids'].append(0)
        if self._isa != impl_spatial_anchor:
            # Update current anchor and check whether this starts a new trajectory
            if self._isa != self._pad_isa and counter['total'] > 0:
                counter['ids'].append(counter['total'])
                is_new_trajectory = True
            self._isa = impl_spatial_anchor

        if counter['total'] > 0 and not is_new_trajectory:
            prev_idx = (idx - 1) % self._memory_size
            ends_buffer[prev_idx] = 0.  # If not a new trajectory, mark previous transition as non-terminal
        ends_buffer[idx] = 1.           # Mark current transition as terminal by default

        # Store transition and action sampling probability
        state, action, reward, next_state = transition['state'], transition['action'], transition['reward'], transition['next_state']
        reward_list = list(reward) if isinstance(reward, (list, tuple)) else [reward]
                                                                # Only supports multi-dimensional rewards as list or tuple
        trans_tensor = torch.tensor(state + [action] + reward_list + next_state, dtype=torch.float32, device=self._device)
        trajs_buffer[idx] = trans_tensor
        probs_buffer[idx] = prob
        
        counter['total'] += 1

    def _get_spatial_context(self):
        self._spatial_context = self._context_decay * self._spatial_context + (1 - self._context_decay) * self._isa
        if abs(self._spatial_context - self._isa) < 1e-3:
            self._spatial_context = self._isa

        return self._spatial_context
    
    def _sample_from_anchor(self, impl_spatial_anchor, batch_size, traj_len = 1):
        '''
        Sample experience data from a specific spatial anchor.

        Args:
            impl_spatial_anchor (int): Target spatial anchor.
            batch_size (int): Number of samples to draw.
            traj_len (int): Length of each trajectory sample.

        Returns:
            states (torch.Tensor): Environment states.
            actions (torch.Tensor): Agent actions.
            rewards (torch.Tensor): Received rewards.
            next_states (torch.Tensor): Next environment states.
            probs (torch.Tensor): Action sampling probabilities.
            masks (torch.Tensor): Masks indicating continuity (1.0 = contiguous, 0.0 = discontinuity).
        '''
        states, actions, rewards, next_states, probs, masks = [], [], [], [], [], []
        trajs_buffer = self._anchor_buffers[impl_spatial_anchor]['trajs']
        probs_buffer = self._anchor_buffers[impl_spatial_anchor]['probs']
        ends_buffer = self._anchor_buffers[impl_spatial_anchor]['ends']

        # Generate sampling indices
        total_num = self._anchor_buffers[impl_spatial_anchor]['counter']['total']
        available_num = min(self._memory_size, total_num)
        if available_num < traj_len or batch_size <= 0: return states, actions, rewards, next_states, probs, masks
        idxs = np.random.choice(available_num, batch_size, replace=batch_size>available_num)

        # Gather trajectory data for the sampled indices
        range_indices = []
        for idx in idxs:
            end_idx = idx + traj_len
            if end_idx > available_num:
                range_indices.extend([[idx, available_num], [0, end_idx % available_num]])
            else:
                range_indices.append([idx, end_idx])
        all_indices = torch.cat([torch.arange(start, end, device=self._device) for start, end in range_indices])
                                                                        # Generate contiguous trajectory indices
        sampled_trajs = trajs_buffer[all_indices]
        sampled_probs = probs_buffer[all_indices]
        sampled_ends = ends_buffer[all_indices]

        # Prepare return tensors
        states = sampled_trajs[:, : self._state_size]
        actions = sampled_trajs[:, self._state_size].long()
        rewards = sampled_trajs[:, self._state_size + 1: self._state_size + 1 + self._reward_size]
        next_states = sampled_trajs[:, -self._state_size :]
        probs = sampled_probs
        masks = 1.0 - sampled_ends  # Terminal transitions become 0, others 1
        masks_indices = torch.arange(traj_len - 1, len(masks), step=traj_len, device=self._device)
        masks[masks_indices] = 0.0  # Ensure last transition of each trajectory has mask 0

        return states, actions, rewards, next_states, probs, masks
    
    def sample(self, batch_size, traj_len = 1):
        '''
        Sample a specified number of trajectory samples.

        Args:
            batch_size (int): Total number of samples to draw.
            traj_len (int): Length of each trajectory sample.

        Returns:
            trajs_num (int): Number of trajectories actually sampled.
            anchors (np.ndarray): Spatial anchors corresponding to samples.
            transitions (dict): Sampled data containing states, actions, rewards, next_states.
            probs (torch.Tensor): Action sampling probabilities for the samples.
            masks (torch.Tensor): Continuity masks (1.0 = contiguous, 0.0 = discontinuity).
        '''
        anchors, states, actions, rewards, next_states, probs, masks = [], [], [], [], [], [], []
        
        # Get smoothed spatial context to guide sampling range
        spatial_context = self._get_spatial_context()

        # Compute contiguous neighboring anchors around current ISA to determine sampling range
        target_spatial = [self._isa]
        for direction in [-1, 1]:           # Iterate both directions along the 1D contiguous anchors
            curr = self._isa + direction
            # curr >= 0 ensures we don't go past the left boundary
            while curr >= 0 and curr in self._visited_anchors:
                target_spatial.append(curr)
                curr += direction
        target_spatial.sort()
        spatial_size = len(target_spatial)  # Total number of neighboring anchors

        # Compute sample counts per neighboring anchor based on spatial context
        trajs_num = 0
        spatial_batchs = get_context_batchs(batch_size, spatial_context - target_spatial[0], spatial_size, self._radius)
        for isa, b_size in zip(target_spatial, spatial_batchs):
            s, a, r, ns, p, m = self._sample_from_anchor(isa, b_size, traj_len)
            sample_num = len(a)
            if sample_num <= 0:
                if isa == target_spatial[-1] and trajs_num == 0:
                    # If the last anchor has no samples, force sampling from the nearest anchor with samples to avoid empty return
                    searcher_order = ripple_traverse(target_spatial, target_spatial.index(self._isa))
                    for searcher_isa in searcher_order:
                        s, a, r, ns, p, m = self._sample_from_anchor(searcher_isa, batch_size, traj_len)
                        sample_num = len(a)
                        if sample_num > 0:
                            isa = searcher_isa
                            b_size = batch_size
                            break
                else:
                    continue
            trajs_num += b_size
            isas = np.full((sample_num,), isa, dtype=int)
            anchors.append(isas)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            probs.append(p)
            masks.append(m)
        
        # Concatenate sampled data from all anchors
        anchors = np.concatenate(anchors, axis=0)
        states = torch.cat(states, dim=0) 
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_states = torch.cat(next_states, dim=0)
        probs = torch.cat(probs, dim=0)
        masks = torch.cat(masks, dim=0)
        transitions = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states
        }
        return trajs_num, anchors, transitions, probs, masks

    def __len__(self):
        '''
        Calculate the number of available experiences within the contiguous spatial range
        determined by the current smoothed spatial context.
        '''
        i, j = math.floor(self._spatial_context), math.floor(self._spatial_context) + 1
        length = 0
        for anchor, direction in [(i, -1), (j, +1)]:
            curr = anchor
            while curr >= 0 and curr in self._visited_anchors:
                length += min(self._memory_size, self._anchor_buffers[curr]['counter']['total'])
                curr += direction
        return length