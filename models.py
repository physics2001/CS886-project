
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv, APPNP, GCNConv
from torch.nn import ModuleList

class APPNPModel(nn.Module): 
    def __init__(self, in_feats, h_feats, num_classes, num_iterations, alpha, dropout):
        super(APPNPModel, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = APPNP(K=num_iterations, alpha=alpha, dropout=dropout)
        self.conv3 = GCNConv(h_feats, num_classes)

    def forward(self, data):
        edge_index = data.edge_index
        x = self.conv1(data.x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)
        

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class MixHopModel(nn.Module): 
    def __init__(self, in_feats, h_feats, num_classes, num_blocks, powers):
        super(MixHopModel, self).__init__()
        self.num_blocks = num_blocks-2
        self.powers = powers
        self.in_hidden_dim = h_feats
        self.out_hidden_dim = h_feats // len(self.powers)
        mixhop_conv_blocks = []
        
        self.conv1 = MixHopConv(in_feats, self.out_hidden_dim, powers=self.powers)
        for i in range(self.num_blocks): 
            mixhop_conv_blocks.append(MixHopConv(self.in_hidden_dim, self.out_hidden_dim, powers=self.powers))
        self.mixhop_conv_blocks = ModuleList(mixhop_conv_blocks)
        self.conv3 = MixHopConv(self.in_hidden_dim, num_classes // 2, powers=[0, 1])

    def forward(self, data):
        edge_index = data.edge_index
        x = self.conv1(data.x, edge_index)
        for i in range(self.num_blocks): 
            x = self.mixhop_conv_blocks[i](x, edge_index)
        x = self.conv3(x, edge_index)
        return x
