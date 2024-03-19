#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:45:32 2024

@author: alien
"""

import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score
from torch_geometric.transforms import RandomLinkSplit
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from DeepGCNModel import DeepGCN
from utils.ckpt_util import load_pretrained_models
from utils.metrics import AverageMeter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import APPNPModel, MixHopModel
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, test_loader, DEVICE, LR, MODEL_TYPE,
                LR_PATIENCE, MAX_EPOCHS, SAVE_PATH, MODEL_CONFIG_STRING, SCHEDULER):
    info_format = 'Epoch: [{}]\t loss: {: .6f} train mF1: {: .6f} \t val mF1: {: .6f}\t test mF1: {:.6f} \t ' \
                  'best val mF1: {: .6f}\t best test mF1: {:.6f}'
    print('===> Init the optimizer ...')
    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=LR_PATIENCE, verbose=True, 
                                  factor=0.5, cooldown=30, min_lr=LR/100)

    print('===> Init Metric ...')
    losses = AverageMeter()

    best_val_value = 0.
    best_test_value = 0.
    
    best_val_acc = 0
    best_test_acc = 0
    
    train_scores = []
    val_scores = []
    
    print('===> Start training ...')
    early_stop_epoch_count = 0
    for epoch in range(MAX_EPOCHS):
        early_stop_epoch_count += 1
        loss, train_value, train_acc = step(model, train_loader, optimizer, criterion, losses, DEVICE)
        val_value, val_acc = test(model, val_loader, DEVICE)
        test_value, test_acc = test(model, test_loader, DEVICE)

        if val_value >= best_val_value:
            best_val_value = val_value
            best_test_value = test_value
            save_ckpt(model, optimizer, scheduler, epoch, SAVE_PATH, 
                      MODEL_CONFIG_STRING, name_post='val_best')
            early_stop_epoch_count = 0
        
        print(info_format.format(epoch, loss, train_value, val_value, test_value, 
                                 best_val_value, best_test_value))
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, train acc: {train_acc:.4f}, " + \
            f"val acc: {val_acc:.4f} (best {best_val_acc:.4f}), " + \
            f"test acc: {test_acc:.4f} (best {best_test_acc:.4f})"
        )
        if SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(losses.avg)
        else:
            scheduler.step()
        
        train_scores.append(train_value)
        val_scores.append(val_value)
        
        if early_stop_epoch_count >= 20: 
            break

    print('Saving the final model.Finish!')
    
    plt.Figure(figsize=(6, 6))
    plt.plot(list(range(len(train_scores))), train_scores, '-b', label='train score')
    plt.plot(list(range(len(val_scores))), val_scores, '-r', label='val score')
    plt.ylim((0, 1))
    plt.xlabel("epoch")
    plt.ylabel("F1 score")
    plt.savefig("plots/train_score_" + MODEL_TYPE + MODEL_CONFIG_STRING + ".png")
    


def step(model, train_loader, optimizer, criterion, losses, DEVICE):
    model.train()
    micro_f1 = 0.
    count = 0.
    acc = 0
    losses.reset()
    for i, data in enumerate(train_loader):
        data = data.to(DEVICE)
        gt = data.y
    
        # ------------------ zero, output, loss
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, gt.float())
        pred = out.argmin(1)

        micro_f1 += f1_score(gt.cpu().detach().numpy(),
                             (out > 0).cpu().detach().numpy(), average='micro') * len(gt)
        count += len(gt)
        # ------------------ optimization
        loss.backward()
        optimizer.step()
        
        acc += (pred.cpu().detach() == gt[:,0].cpu().detach()).float().sum()

        losses.update(loss.item())
    return losses.avg, micro_f1/count, acc/count


def test(model, loader, DEVICE):
    model.eval()
    count = 0
    micro_f1 = 0.
    acc = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(DEVICE)
            out = model(data)
            pred = out.argmin(1)

            num_node = len(data.x)
            micro_f1 += f1_score(data.y.cpu().detach().numpy(),
                                 (out > 0).cpu().detach().numpy(), average='micro') * num_node
            
            acc += (pred.cpu().detach() == data.y[:,0].cpu().detach()).float().sum()
            
            count += num_node
        micro_f1 = float(micro_f1)/count
        acc /= count
    return micro_f1, acc


def save_ckpt(model, optimizer, scheduler, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)


def run_pipeline(LR_PATIENCE, LR, SCHEDULER, BATCH_SIZE, DEVICE, 
                 MAX_EPOCHS, DATA_FOLDER, MODEL_TYPE, SAVE_PATH, 
                 MODEL_CONFIG): 
    print('===> Creating dataloader ...')
    github_data = GeoData.GitHub(root=DATA_FOLDER)
    github_data.data.y = torch.stack((github_data.data.y, 1-github_data.data.y), 1).float()
    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(github_data[0])
    train_loader = NeighborLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[30]*2)
    val_loader = NeighborLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[30]*2)
    test_loader = NeighborLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[30]*2)
    n_classes = github_data.num_classes
    
    MODEL_CONFIG_STRING = str(MODEL_CONFIG)
    
    print('===> Loading the network ...')
    if MODEL_TYPE == "DeepGCN": 
        print('===> Initializing DeepGCN ...')
        n_filters = MODEL_CONFIG["n_filters"]
        n_blocks = MODEL_CONFIG["n_blocks"]
        conv = MODEL_CONFIG["conv"]
        dropout = MODEL_CONFIG["dropout"]
        model = DeepGCN(in_channels=github_data.num_features, n_classes=n_classes, 
                        n_filters=n_filters, act='relu', norm='batch', bias=True, conv=conv, 
                        n_heads=1, n_blocks=n_blocks, dropout=dropout).to(DEVICE)
    elif MODEL_TYPE == "APPNP": 
        print('===> Initializing APPNP ...')
        h_feats = MODEL_CONFIG["h_feats"]
        num_iterations = MODEL_CONFIG["num_iterations"]
        alpha = MODEL_CONFIG["alpha"]
        dropout = MODEL_CONFIG["dropout"]
        model = APPNPModel(in_feats=github_data.num_features, h_feats=h_feats, num_classes=n_classes,
                           num_iterations=num_iterations, alpha=alpha, dropout=dropout).to(DEVICE)
    elif MODEL_TYPE == "MixHop": 
        print('===> Initializing MixHop ...')
        h_feats = MODEL_CONFIG["h_feats"]
        n_blocks = MODEL_CONFIG["n_blocks"]
        powers = MODEL_CONFIG["powers"]
        model = MixHopModel(in_feats=github_data.num_features, h_feats=h_feats,
                            num_classes=n_classes, num_blocks=n_blocks, powers=powers).to(DEVICE)
    
    train_model(model, train_loader, val_loader, test_loader, DEVICE, LR, MODEL_TYPE,
                LR_PATIENCE, MAX_EPOCHS, SAVE_PATH, MODEL_CONFIG_STRING, SCHEDULER)

    # load best model on validation dataset
    print('\n\n=================Below is best model testing=====================')
    print('===> loading pre-trained ...')
    best_model_path = '{}/{}_val_best.pth'.format(SAVE_PATH, MODEL_CONFIG_STRING)
    model, best_value, epoch = load_pretrained_models(model, best_model_path, "test")
    val_value, val_acc = test(model, val_loader, DEVICE)
    print('Val m-F1: {: 6f}'.format(val_value))
    print('Val acc: {: 6f}'.format(val_acc.item()))
    test_value, test_acc = test(model, test_loader, DEVICE)
    print('Test m-F1: {: 6f}'.format(test_value))
    print('Test acc: {: 6f}'.format(test_acc.item()))
    
    return test_value, test_acc.item()
        
        
        