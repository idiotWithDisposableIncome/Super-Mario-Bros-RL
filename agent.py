import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torch.optim.lr_scheduler import StepLR

class Agent:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.000275, 
                 gamma=0.9, 
                 epsilon=1.0, 
                 eps_decay=0.99999975, 
                 eps_min=0.1, 
                 replay_buffer_capacity=100_000, 
                 batch_size=32, 
                 sync_network_rate=25000):
        
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=20000, gamma=0.9999)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss() # Feel free to try this loss function instead!

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Passing in a list of numpy arrays is slower than creating a tensor from a numpy array
        # Hence the `np.array(observation)` instead of `observation`
        # observation is a LIST of numpy arrays because of the LazyFrame wrapper
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state_tensor, action, reward, next_state_tensor, done):
        self.replay_buffer.add(TensorDict({
            "state": state_tensor, 
            "action": torch.tensor(action, dtype=torch.int64),
            "reward": torch.tensor(reward, dtype=torch.float32), 
            "next_state": next_state_tensor, 
            "done": torch.tensor(done, dtype=torch.bool)
        }, batch_size=[]))
        
    def handle_experiences(self, experiences):
        for state, action, reward, next_state, done in experiences:
            # Convert state and next_state to PyTorch tensors
            state_tensor = self._prepare_state(state)
            next_state_tensor = self._prepare_state(next_state)
            
            self.store_in_memory(state_tensor, action, reward, next_state_tensor, done)
            self.learn()

    def _prepare_state(self, state):

        if isinstance(state, tuple):
            state = state[0]  # If state is a tuple, extract the actual state
        # Convert LazyFrames to numpy, reshape, and convert to tensor
        state_np = np.array(state, dtype=np.float32) / 255.0  # Convert LazyFrames to numpy array
        #print(f"Shape after conversion to numpy: {state_np.shape}")  # Debug print
        tensor = torch.tensor(state_np, dtype=torch.float32).to(self.online_network.device)
        #state_np = state_np.transpose((2, 0, 1))  # Reshape to [channels * num_stacks, height, width]
        #print(f"Shape after transpose: {state_np.shape}")  # Debug print
        #tensor = torch.tensor(state_np).unsqueeze(0).to(self.online_network.device)
        #print(f"Shape after tensor conversion: {tensor.shape}")  # Debug print
        return tensor
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        #print(f"Replay buffer size before sampling: {len(self.replay_buffer)}")  # Debug print
        self.sync_networks()
        
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        states = states.squeeze(1)
        #print(f"Shape of batched states: {states.shape}")  # Debug print

        predicted_q_values = self.online_network(states) # Shape is (batch_size, n_actions)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # The rewards of any future states don't matter if the current state is a terminal state
        # If done is true, then 1 - done is 0, so the part after the plus sign (representing the future rewards) is 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

        self.scheduler.step()


        


