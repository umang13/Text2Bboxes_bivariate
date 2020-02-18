import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.autograd import Variable
import numpy as np
from MDN import MDN

class Text2BBoxesModel(nn.Module) :
    def __init__(self, hidden_size, labels, batch_size,device) :
        super(Text2BBoxesModel, self).__init__()
        self.device = device
        self.num_categories = labels[max(labels, key=labels.get)] + 1
        # input size is num_classes + bbox_coordinates
        self.lstm = nn.LSTMCell(self.num_categories + 4, hidden_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.output_label = nn.Linear(hidden_size, self.num_categories)
        self.output_coords = nn.Linear(hidden_size + self.num_categories, 2)
        self.output_size = nn.Linear(hidden_size + self.num_categories + 2, 2)
        self.batch_size = batch_size
        self.SOS = self.num_categories - 2
        self.EOS = self.num_categories - 1
        self.num_mixtures = 20
        self.mdn_xy_model = MDN(hidden_size + self.num_categories, self.num_mixtures, 2).to(device)
        self.mdn_wh_model = MDN(hidden_size + self.num_categories + 2, self.num_mixtures, 2).to(device)
        self.ONEOVERSQRT2PI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians
        
       # self.output_coords = nn.Linear(hidden_size)
    
        
    def init_hidden_states(self, input_embedding) :
        self.hidden = input_embedding.to(self.device)
        self.cell = torch.zeros_like(input_embedding, requires_grad=True).to(self.device)

    def forward(self, caption_embedding, max_length) :

        self.init_hidden_states(caption_embedding)
        t = 0
        start_tensor = self.index_to_one_hot(self.SOS)
        pred_labels = start_tensor.repeat(self.batch_size,1).float()
        pred_labels.requires_grad_(True)
        bbox_tensor = torch.Tensor([0,0,0,0])
        bbox_tensor = bbox_tensor.repeat(self.batch_size, 1)
        bbox_tensor.requires_grad_(True)
        pred_labels = pred_labels.to(self.device)
        bboxes = bbox_tensor.to(self.device)
        outputs = []
        while(t < max_length) :
            inputs = torch.cat([pred_labels, bboxes], 1)
            self.hidden, self.cell = self.lstm(inputs, (self.hidden, self.cell))
            self.hidden = self.dropout_layer(self.hidden)
            pred_labels_logits = self.output_label(self.hidden)
            pred_labels = self.softmax(pred_labels_logits)
            theta_xy = self.mdn_xy_model(torch.cat([self.hidden, pred_labels_logits], 1))
            sample_xy = self.sample_from_gaussian(theta_xy).to(self.device)
            # sample_xy = self.sample_from_theta(print(theta_xy)
            # print(sample_xy, theta_xy[0] * self.gaussian_probability(theta_xy[1], theta_xy[2], sample_xy))
            # print(sample_xy.shape)
            theta_wh = self.mdn_wh_model(torch.cat([self.hidden, pred_labels_logits, sample_xy], 1))
            # sample_wh = theta_wh[2].squeeze(1)
            sample_wh = self.sample_from_gaussian(theta_wh).to(self.device)
            bboxes = torch.cat([sample_xy, sample_wh], 1)
            t+=1
            output = [pred_labels, theta_xy, theta_wh]
            outputs.append(output)
        return outputs
    
    def loss_function_labels(self, preds, labels) :
        return F.cross_entropy(preds, labels)
    
    def one_hot_to_label(self, label) :
        value, index = torch.max(label[0], 0)
        return index

    def index_to_one_hot(self, index) :
        labels_one_hot = torch.Tensor(self.num_categories).zero_()
        labels_one_hot[index] = 1
        return labels_one_hot

    def gaussian_probability(self, sigma, mu, target):
        """Returns the probability of `target` given  parameters `sigma` and `mu`.
        
        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. 
            mu (BxGxO): The means of the Gaussians. 
            data (BxI): A batch of data.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        eps = 1e-20
        target = target.unsqueeze(1).expand_as(sigma)
        ret = self.ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / (sigma + eps))**2) / (sigma + eps)
        return torch.prod(ret, 2)

    def weighted_logsumexp(self, x,w, dim=None, keepdim=False):
        
        eps = 1e-20
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            (xm == float('inf')) | (xm == float('-inf')), # handle inf cases that produce nan's
            xm,
            xm + torch.log(torch.sum(torch.exp(x - xm)*w.unsqueeze(2), dim, keepdim=True) + eps)) 
        return x if keepdim else x.squeeze(dim)

    def log_sum_exp(self,value, weights, dim=None):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        eps = 1e-20
        m, idx = torch.max(value, dim=dim, keepdim=True)
        # print("==>", (value.shape,value[0,:,:]-m[0,:,:]).float())
        # print("^^^", m.shape, m[0,:,:].float())
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(value-m + torch.log(weights.unsqueeze(2)+eps)),
                                       dim=dim) + eps)
        

    def mdn_loss(self, pi, sigma, mu, target):
        """Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        eps = 1e-20
        target = target.unsqueeze(1)
        # print(target, target.shape)
        # print(sigma)
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        probs = m.log_prob(target)

        # print(probs.shape)
        # # print(probs.max(dim=1))
        # loss = torch.exp(probs)
        # # print(loss.shape, pi.unsqueeze(2).shape)
        # loss = torch.sum(loss * pi.unsqueeze(2), dim=1)
        # loss = -torch.log(loss + eps)
        # print(loss.mean(), -self.weighted_logsumexp(probs, pi, dim=1).mean())
        loss = -self.log_sum_exp(probs, pi, dim=1)
        return loss.mean()

    def sample_from_theta(self, theta) :
        pi, sigma, mu = theta
        categorical = Categorical(pi)
        pis = list(categorical.sample().data)
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
        for i, idx in enumerate(pis):
            sample[i] = sample[i].clone().mul(sigma[i,idx]).add(mu[i,idx])

        #print("####", sample.shape)
        sample = sample.to(self.device)
        # pi , sigma, mu = pi.to(self.device), sigma.to(self.device), mu.to(self.device)
        return sample

    def sample_max_probs_from_theta(self, theta) :
        pi, sigma, mu = theta
        return torch.mean(torch.mul(mu, pi.unsqueeze(2)), 1)

    def gumbel_sample(self, data) :
        """ We use gumbel sampling to pick on the mixtures based on their mixture 
        weights """
        distribution = Gumbel(loc=0, scale=1)
        z = distribution.sample()
        # z = np.random.gumbel(loc=0, scale=1, size=data.shape)
        return (torch.log(data) + z).argmax(dim=1)

    def sample_from_gaussian(self, theta) :
        pi, sigma, mu = theta
        k = self.gumbel_sample(pi)
        select_mu = torch.zeros(mu.shape[0], mu.shape[2])
        select_sigma = torch.zeros(sigma.shape[0], sigma.shape[2])
        # print(select_mu.shape, select_sigma.shape, k.shape, k)
        for idx, i in enumerate(k) :
            select_mu[idx,:] = mu[idx,i,:]
            select_sigma[idx,:] = sigma[idx,i,:]
        distribution = Normal(select_mu, select_sigma)
        return distribution.sample()
