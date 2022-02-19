# Inspired from https://github.com/raillab/dqn
from gym import spaces
import numpy as np

# from dqn.model_nature import DQN as DQN_nature
# from dqn.model_neurips import DQN as DQN_neurips
from dqn.model_moco import DQN, MomentumEncoder
from dqn.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F


class PytorchQueue:
    def __init__(self, K, device):
        self.K = K
        self.device = device
        self.queue = torch.tensor([], dtype=torch.float32, device=device)

    def enqueue(self, tensor):
        self.queue = torch.cat((self.queue, tensor), 0)
    
    def dequeue(self):
        elements_to_remove = max(len(self.queue-self.K), 0)
        self.queue = self.queue[elements_to_remove : ]
    
    def view(self, *args):
        return self.queue.view(*args)
    
    def reached_max_capacity(self):
        return (len(self.queue) >= self.K)


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 use_double_dqn,
                 lr,
                 batch_size,
                 gamma,
                 device=torch.device("cpu" ),
                 dqn_type="neurips"):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        print(f'Using: {device}')

        # if(dqn_type=="neurips"):
        #     DQN = DQN_neurips
        # else:
        #     DQN = DQN_nature

        # DQN = DQN_nature

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.momentum_encoder = MomentumEncoder(observation_space).to(device)
        self.momentum_encoder.set_default_params(self.policy_network)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.queue = PytorchQueue(1024, device=device)
        self.update_target_network()
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters()
            , lr=lr)        
        ## self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.device = device

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.policy_network(next_states)[0].max(1)
                max_next_q_values = self.target_network(next_states)[0].gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_states)[0]
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        self.policy_network.calculate_representation_loss=True
        input_q_values, query_key = self.policy_network(states)
        self.policy_network.calculate_representation_loss=False
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values) + self.representation_loss(query_key)
        self.momentum_encoder.set_params(self.policy_network.encoder.parameters())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()
    
    def representation_loss(self, query_key):
        K = 64
        T = 0.07
        q, x_k = query_key
        N, C = q.shape
        k = self.momentum_encoder(x_k)

        # No training on representation loss until the queue is full
        if self.queue.reached_max_capacity():
            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
            l_neg = l_neg = torch.mm(q.view(N,C), self.queue.view(C,K))

            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(N)
            loss = F.CrossEntropyLoss(logits/T, labels)
        else:
            loss = torch.tensor([0.], device=self.device)

        self.queue.enqueue(k)
        self.queue.dequeue()

        return loss


    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)[0]
            _, action = q_values.max(1)
            return action.item()
