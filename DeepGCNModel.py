import torch
from torch.nn import Linear as Lin, Sequential as Seq
from DeepGCNVertices import GraphConv, ResGraphBlock
from DeepGCNLayers import MultiSeq, MLP


class DeepGCN(torch.nn.Module):
    def __init__(self, in_channels, n_classes, n_filters, act, norm, bias, conv, n_heads, n_blocks, dropout):
        super(DeepGCN, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.channels = n_filters
        self.act = act
        self.norm = norm
        self.bias = bias
        self.conv = conv
        self.heads = n_heads
        self.c_growth = 0
        self.n_blocks = n_blocks
        self.head = GraphConv(self.in_channels, self.channels, self.conv, 
                              self.act, self.norm, self.bias, self.heads)
        self.dropout = dropout
        self.res_scale = 1
        self.backbone = MultiSeq(*[ResGraphBlock(self.channels, self.conv, 
                                                 self.act, self.norm, self.bias, 
                                                 self.heads, self.res_scale)
                                   for _ in range(self.n_blocks-1)])
        self.fusion_dims = int(self.channels * self.n_blocks + self.c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))
        self.fusion_block = MLP([self.fusion_dims, 1024], self.act, None, self.bias)
        self.prediction = Seq(*[MLP([1+self.fusion_dims, 512], self.act, self.norm, self.bias), 
                                torch.nn.Dropout(p=self.dropout),
                                MLP([512, 256], self.act, self.norm, self.bias), 
                                torch.nn.Dropout(p=self.dropout),
                                MLP([256, self.n_classes], None, None, self.bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        feats = [self.head(x, edge_index)]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], edge_index)[0])
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)
        out = self.prediction(torch.cat((feats, fusion), 1))
        return out
