import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MDN(nn.Module) :
        def __init__(self, input_size, num_mixtures, out_features) :
                super(MDN, self).__init__()
                self.input_size = input_size
                self.out_features = out_features
                self.num_mixtures = num_mixtures
                # self.mdn_param_layer =
                self.pi_layer = nn.Linear(self.input_size, self.num_mixtures)
                self.sigma_layer = nn.Linear(self.input_size, self.num_mixtures*self.out_features)
                self.mu_layer = nn.Linear(self.input_size, self.num_mixtures*self.out_features)

        def forward(self, input) :
                pi = F.softmax(self.pi_layer(input), 1)
                # print('nan values:', torch.sum(torch.isnan(out_probs)).item())
                sigma = torch.exp(self.sigma_layer(input))
                # print(torch.le(sigma , 1e-10)==True)
                # intmd = torch.clamp(intmd, min=0, max=15)
                # sigma = torch.exp(sigma)
                mu = self.mu_layer(input)
                sigma = sigma.view(-1, self.num_mixtures, self.out_features)
                mu = mu.view(-1, self.num_mixtures, self.out_features)
                # print(mu.shape)
                return [pi, sigma, mu]
