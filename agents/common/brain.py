import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from infra.utils.tools import get_autocast_dtype

class DNN(nn.Module):
    def __init__(self, state_dims, n_actions):
        super(DNN, self).__init__()
        self._layer_1 = nn.Linear(state_dims, 64)
        self._layer_2 = nn.Linear(64, 64)
        
        self._layer_3 = nn.Linear(64, 64)
        self._layer_4 = nn.Linear(64, 64)

        self._layer_5 = nn.Linear(64, 64)
        self._layer_6 = nn.Linear(64, 64)

        self._action_head = nn.Linear(64, n_actions)

    def forward(self, state):
        h1 = F.relu(self._layer_1(state))
        h2 = F.relu(self._layer_2(h1))

        h3 = F.relu(self._layer_3(h2))
        h4 = F.relu(self._layer_4(h3))  # Residual Block

        add1 = torch.add(h2, h4)

        h5 = F.relu(self._layer_5(add1))
        h6 = F.relu(self._layer_6(h5))  # Residual Block

        shared_features = torch.add(add1, h6)

        action_outputs = self._action_head(shared_features)

        return action_outputs

class DQN():
    def __init__(
            self, state_dims, n_actions, n_nodes,
            replace_target_iter = 200, learning_rate = 1e-3, gamma = 0.9,
            epsilon = 1., epsilon_min = 0.01, epsilon_decay = 0.995,
            device = 'cuda'
        ):
        # copy params
        self._state_dims = state_dims
        self._n_actions = n_actions
        self._n_nodes = n_nodes
        self._rt_iter = replace_target_iter
        self._lr = learning_rate
        self._gamma = gamma
        self._eps = epsilon
        self._eps_min = epsilon_min
        self._eps_decay = epsilon_decay

        # create policy network and target network
        self._device_name = device.lower()
        if self._device_name != 'cpu': assert torch.cuda.is_available(), "[Error] CUDA is unavailable!"
        self._amp_dtype = get_autocast_dtype(self._device_name)
        self._net_device = torch.device(device)
        self._model = DNN(self._state_dims, self._n_actions).to(self._net_device)
        total_params = sum(p.numel() for p in self._model.parameters())
        print("[Info] DQN using DNN as the decision network. Total number of parameters: {}.".format(total_params))
        self._target_model = copy.deepcopy(self._model)
        self._target_model.eval()
        self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr = self._lr)
        self._model = torch.compile(self._model, mode="reduce-overhead")
        self._target_model = torch.compile(self._target_model, mode="reduce-overhead")
        self._amp_scaler = GradScaler()
        self._learn_step = 0
        self._t = 0

    def choose_action(self, state):
        self._model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self._device_name, dtype=self._amp_dtype):
                sa_v = self._model(state)

        if np.random.random() < self._eps:
            actions = torch.randint(0, self._n_actions, (self._n_nodes,),
                                    device=self._net_device, dtype=torch.int)
        else:
            _, optimal_act = torch.max(sa_v, dim=1)
            actions = optimal_act
        self._eps = max(self._eps_min, self._eps * self._eps_decay)
        return actions, sa_v

    def _replace_target_params(self):
        weights = self._model.state_dict()
        self._target_model.load_state_dict(weights)

    def learn(self, states, actions, rewards, next_states):
        self._model.train()
        self._learn_step += 1
        batch_size = states.size(0)
        batch_index = torch.arange(batch_size, device=self._device_name, dtype=torch.long)
        with autocast(device_type=self._device_name, dtype=self._amp_dtype): # Automatic Mixed Precision (AMP)
            q_eval = self._model(states)
            current_q = q_eval[batch_index, actions].view(-1,1)
            with torch.no_grad():
                q_next = self._target_model(next_states)
                max_next_q = q_next.max(1)[0].view(-1, 1)
                target_q = rewards + self._gamma * max_next_q
            q_loss = F.mse_loss(current_q, target_q)
        
        self._optimizer.zero_grad(set_to_none=True)
        if self._device_name == 'cpu':
            q_loss.backward()
            self._optimizer.step()
        else:
            self._amp_scaler.scale(q_loss).backward()
            self._amp_scaler.step(self._optimizer)
            self._amp_scaler.update()

        if self._learn_step % self._rt_iter == 0:
            self._replace_target_params()

        return {'RL_loss':q_loss.item()}
    
    def save(self, save_path):
        # Check if the model is a compiled object (torch.compile)
        if hasattr(self._model, "_orig_mod"):
            model_state = self._model._orig_mod.state_dict()
        else:
            model_state = self._model.state_dict()
        if hasattr(self._target_model, "_orig_mod"):
            target_model_state = self._target_model._orig_mod.state_dict()
        else:
            target_model_state = self._target_model.state_dict()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model_state,
            'target_model_state_dict': target_model_state,
            'optimizer_state_dict': self._optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)