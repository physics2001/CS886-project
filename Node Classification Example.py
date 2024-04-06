import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import APPNP, GCN, MixHop

from torch_geometric.loader import DataLoader, LinkLoader, NeighborLoader
from torch_geometric.sampler import EdgeSamplerInput
from torch_geometric.nn.data_parallel import DataParallel
from sklearn.metrics import f1_score
from torch_geometric.transforms import RandomLinkSplit, AddSelfLoops, RemoveIsolatedNodes
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataListLoader, Dataset

DATA_FOLDER = "data/Github"
BATCH_SIZE = 512
# dataset = dgl.data.CoraGraphDataset()
# print(f"Number of categories: {dataset.num_classes}")

# g = dataset[0]
# g.ndata['label'].shape

# print("Node features")
# print(g.ndata)
# print("Edge features")
# print(g.edata)

github_data = GeoData.GitHub(root=DATA_FOLDER)
transform = RandomLinkSplit(is_undirected=True)
# removeIsolatedNodes = RemoveIsolatedNodes()
train_data, val_data, test_data = transform(github_data[0])
# train_data = removeIsolatedNodes(train_data.data)
# val_data = removeIsolatedNodes(val_data.data)
# test_data = removeIsolatedNodes(test_data.data)
print(github_data.data)
print(train_data)
print(test_data)
print(val_data)

test_loader = NeighborLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[30]*2)
n_classes = github_data.num_classes


# Create the model with given dimensions
# model = APPNP(g.ndata["feat"].shape[1], 64, dataset.num_classes, 10, 0.1)
# model = MixHop(g.ndata["feat"].shape[1], 128, dataset.num_classes, 20)
model = APPNP(github_data.num_features, 128, n_classes, 12, 0.1).to('cuda')


g_train = dgl.graph(data=(train_data.edge_label_index[0], train_data.edge_label_index[1]), num_nodes=37700)
g_train = dgl.to_bidirected(g_train)
g_train.ndata["feat"] = train_data.x
g_train.ndata["label"] = train_data.y
g_train = dgl.add_self_loop(g_train)

g_test = dgl.graph(data=(test_data.edge_label_index[0], test_data.edge_label_index[1]), num_nodes=37700)
g_test = dgl.to_bidirected(g_test)
g_test.ndata["feat"] = test_data.x
g_test.ndata["label"] = test_data.y
g_test = dgl.add_self_loop(g_test)

g_val = dgl.graph(data=(val_data.edge_label_index[0], val_data.edge_label_index[1]), num_nodes=37700)
g_val = dgl.to_bidirected(g_val)
g_val.ndata["feat"] = val_data.x
g_val.ndata["label"] = val_data.y
g_val = dgl.add_self_loop(g_val)


def train(g_train, g_test, g_val, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    for e in range(200):
        # Forward
        train_logits = model(g_train, g_train.ndata["feat"])

        # Compute prediction
        train_pred = train_logits.argmax(1)
        
        test_logits = model(g_test, g_test.ndata["feat"])

        # Compute prediction
        test_pred = test_logits.argmax(1)
        
        val_logits = model(g_val, g_val.ndata["feat"])

        # Compute prediction
        val_pred = val_logits.argmax(1)
        
        # print(logits.shape)
        # print(labels.shape)
        # print(train_mask.shape)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = nn.NLLLoss()(logits[train_mask[:, 0]], labels[train_mask[:, 0]])
        loss = F.cross_entropy(train_logits, g_train.ndata["label"])

        # Compute accuracy on training/validation/test
        train_acc = (train_pred == g_train.ndata["label"]).float().mean()
        val_acc = (val_pred == g_val.ndata["label"]).float().mean()
        test_acc = (test_pred == g_test.ndata["label"]).float().mean()


        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 20 == 19:
            print(
                f"In epoch {e}, loss: {loss:.3f}, train acc: {train_acc:.4f}, " + \
                f"val acc: {val_acc:.4f} (best {best_val_acc:.4f}), " + \
                f"test acc: {test_acc:.4f} (best {best_test_acc:.4f})"
            )

g_train = g_train.to('cuda')
g_test = g_test.to('cuda')
g_val = g_val.to('cuda')
train(g_train, g_test, g_val, model)
