from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU()
            # nn.Linear(in_features=512, out_features=action_space.n)
        )

        self.policy_fc = nn.Linear(in_features=512, out_features=action_space.n)
        
        self.decoder= nn.Sequential(
            nn.Linear(in_features=512, out_features=64*7*7),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=observation_space.shape[0], kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0],-1)
        encoder_out = self.encoder_fc(conv_out) 

        policy = self.policy_fc(encoder_out)            

        decoder_out = self.decoder(encoder_out)

        return policy, decoder_out
