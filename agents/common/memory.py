import torch

class ReplayMemory:
    def __init__(self, memory_size, state_size, device: str):
        self._memory_size = memory_size
        self._state_size = state_size
        self._device = device.lower()
        self.reset()

    def reset(self):
        self._memory = torch.zeros((self._memory_size, self._state_size * 2 + 2), device=self._device)
        self._memory_counter = 0

    def store_transition(self, state, action, reward, next_state):
        combined = list(state) + [action, reward] + list(next_state)
        trans_tensor = torch.tensor(combined, dtype=torch.float32, device=self._device)
        
        idx = self._memory_counter % self._memory_size
        self._memory[idx] = trans_tensor
        self._memory_counter += 1

    def sample(self, batch_size):
        current_count = min(self._memory_counter, self._memory_size)
        idxs = torch.randperm(current_count, device=self._device)[:batch_size]

        samples = self._memory[idxs, :]
        states = samples[:, :self._state_size]
        actions = samples[:, self._state_size].long()
        rewards = samples[:, self._state_size + 1].unsqueeze(-1)    # (batch_size, 1)
        next_states = samples[:, -self._state_size:]

        return states, actions, rewards, next_states