import os, ipdb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt

class MonotonicNetwork(nn.Module):
    def __init__(self, n_nodes, init='rand'):
        super(MonotonicNetwork, self).__init__()
        self.n_nodes = n_nodes
        if init == 'linear':
            shifts_init = torch.linspace(0, 1, n_nodes)
        elif init == 'rand':
            shifts_init = torch.rand(n_nodes)
            # shifts_init = torch.randn(n_nodes)*0.5 + .5
        else:
            raise ValueError()

        shifts_init.clamp_(0, 1)
        self.shifts = nn.Parameter(shifts_init)
        self.scales = nn.Parameter(torch.ones(n_nodes) * 15)

    def network_pass(self, x):
        # Only positive shifts/scales
        shifts = torch.relu(self.shifts)
        scales = torch.relu(self.scales)
        x = x.expand(-1, self.n_nodes)
        z = scales * (x - shifts)
        y = torch.sigmoid(z).mean(-1, keepdim=True)
        return y 

    def forward(self, x):       
        # Input pass
        y = self.network_pass(x)
        # Scale things to be in [0, 1]
        zeros = self.network_pass(torch.zeros_like(x))
        ones = self.network_pass(torch.ones_like(x))
        y = (y - zeros) / (ones - zeros + 1e-6)
        return y


if __name__ == '__main__':
    
    os.makedirs('_monotonicnet', exist_ok=True)

    x= torch.linspace(-1,2,1000).unsqueeze(1)
    for init in ['linear', 'rand']:
        fig = plt.subplots(1, 3, figsize=(10,3), sharey=True, sharex=True)
        for i, n_nodes in enumerate([5, 10, 50]):
            plt.subplot(1, 3, i+1)
            for j in range(30):
                monotonic_net = MonotonicNetwork(n_nodes, init)
                y = monotonic_net(x).detach()
                if j == 0:
                    plt.plot(x, y, c=f'C{i}', alpha=0.3, label=f'K={n_nodes}')
                else:
                    plt.plot(x, y, c=f'C{i}', alpha=0.3)
            plt.legend()

        plt.ylim([-.0,1.])
        plt.xlim([-.0,1.])
        plt.subplot(1, 3, 2)
        plt.xlabel('Input Phase', fontsize=12)
        plt.subplot(1, 3, 1)
        plt.ylabel('Output Phase', fontsize=12)
        plt.savefig(f'_monotonicnet/n_nodes_{init}.png', bbox_inches='tight')