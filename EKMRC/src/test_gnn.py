#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_gnn.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/04/22 15:19:24
'''

# here put the import lib
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

edge_index = torch.tensor([[0, 2],
                           [2, 0],
                           [3, 2],
                           [2, 3]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)


data = Data(x=x, edge_index=edge_index.t().contiguous())
device = torch.device('cuda')
data = data.to(device)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    # optimizer.zero_grad()
    out = model(data)