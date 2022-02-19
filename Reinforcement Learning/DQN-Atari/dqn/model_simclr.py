from gym import spaces
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # print(sim.shape)

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

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
        
        # self.decoder= nn.Sequential(
        #     nn.Linear(in_features=512, out_features=64*7*7),
        #     nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=observation_space.shape[0], kernel_size=8, stride=4),
        #     nn.ReLU(),
        # )

        self.projection = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.similarity = nn.CosineSimilarity()
        self.calculate_representation_loss = False
    
    def encoder(self, x):
        conv_out = self.conv(x).view(x.size()[0],-1)
        encoder_out = self.encoder_fc(conv_out)
        return encoder_out

    def forward(self, x):
        encoder_out = self.encoder(x)
        policy = self.policy_fc(encoder_out)

        # decoder_out = self.decoder(encoder_out)
        self_supervised_loss = None
        if self.calculate_representation_loss:
            self_supervised_loss = self.calculate_self_supervised_loss(x)

        return policy, self_supervised_loss
    

    def calculate_self_supervised_loss(self, x):
        N = x.shape[0]
        z = []
        # for x_k in x:
        #     # First augmentation
        #     x_tilda = augmentation(x_k.unsqueeze(0))
        #     h = self.encoder(x_tilda)
        #     z_k = self.projection(h)
        #     z.append(z_k)
            
        #     # Second augmentation
        #     x_tilda = augmentation(x_k.unsqueeze(0))
        #     h = self.encoder(x_tilda)
        #     z_k = self.projection(h)
        #     z.append(z_k)

        z1 = self.projection(self.encoder(augmentation(x)))
        z2 = self.projection(self.encoder(augmentation(x)))
        
        # s = [[None for _ in range(2*N)] for _ in range(2*N)]
        # for i in range(2*N):
        #     for j in range(2*N):
        #         s[i][j] = self.similarity(z[i], z[j])
        
        # loss = l(0, 1, s) + l(1, 0, s)
        # for k in range(1, N):
        #     loss = loss + l(2*k, 2*k+1, s) + l(2*k+1, 2*k, s)
        # loss /= (2*N)

        # return loss
        # t1 = torch.tensor([1.], requires_grad=True)
        # t2 = torch.tensor([1.])
        # return torch.dot(t1, t2)

        criterion = SimCLR_Loss(x.shape[0], 1)
        loss = criterion(z1, z2)
        return loss


# Stochastic data augmentation
def augmentation(x):
    H, W = x.shape[-2], x.shape[-1]
    transforms = T.Compose([
        T.RandomCrop((H-3, W-2)),
        T.Resize((H, W)),
        T.GaussianBlur(3)
    ])
    return transforms(x)

def l(i:int, j:int, s, temperature=1):
    start_idx = 1 if (i==0) else 0
    denominator = torch.exp(s[i][start_idx]/temperature)
    for k in range(start_idx, len(s)):
        if k == i:
            continue
        denominator = denominator + torch.exp(s[i][k]/temperature)
    return torch.log(denominator) - (s[i][j]/temperature)