"""
https://pytorch.org/tutorials/intermediate/dist_tuto.html
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import numpy as np
import os
import time
import argparse
from argparse import Namespace

from tqdm import tqdm

from circuit.qnn_torch_pure import QNN

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--W", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--port", type=int, default=0)
    opt = parser.parse_args()
    return opt

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

""" Weight averaging. """
def average_weights(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def step(model, feat, label, optimizer, K):
    local_batch_size = len(feat) // K
    for i in range(0, len(feat), local_batch_size):
        loss = 0
        for j in range(local_batch_size):
            loss += (model(feat[i+j]) - label[i+j])**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, feat_test, label_test):
    n_correct = 0
    feat_test = torch.from_numpy(feat_test.astype(np.complex64))
    for i in range(len(feat_test)):
        predict = model(feat_test[i])
        if (predict.item()-0.5) * label_test[i] > 0:
            n_correct += 1
    return n_correct / len(feat_test)

def train(model, feat, label, feat_test=None, label_test=None, batch_size=4, local_iter=8, p=0.1, M=100):
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': 1e-2, 'weight_decay': 0}], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.5, last_epoch=-1)

    np.random.seed(0)
    loss_list = []
    grad_list = []
    for i in range(100):
        index = np.random.permutation(len(feat))
        feat, label = feat[index], label[index]
        loss = 0
        for j in range(0, len(feat), batch_size):
            feat_batch = torch.from_numpy(feat[j:j+batch_size].astype(np.complex64))
            label_batch = torch.from_numpy(label[j:j+batch_size]).long().to(model.device)
            loss = 0
            local_batch_size = len(feat_batch) // local_iter + (len(feat_batch) % local_iter != 0)

            #cnt = 0
            #st = time.time()
            for k in range(0, len(feat_batch), local_batch_size):
                loss = 0
                for m in range(k, min(k+local_batch_size, len(feat_batch))):
                    predict = model(feat_batch[m])
                    predict = (predict - 0.5)*2
                    loss += (label_batch[m] - predict)**2
                    #cnt += 1
                loss = loss / (min(k+local_batch_size, len(feat_batch)) - k)
                optimizer.zero_grad()
                loss.backward()
                grad_list.append(np.linalg.norm([param.grad.numpy() for param in model.parameters()][0].flatten()))
                optimizer.step()
            #if dist.get_rank() == 0:
            #    print('Batch: {}, size: {}'.format(time.time() - st, cnt))
            #average_gradients(model)
            # average_weights(model)
            #if dist.get_rank() == 0:
            #    print('After sync: {}'.format(time.time() - st))
        scheduler.step()
        loss_list.append(loss.item())

        if dist.get_rank() == 0:
            print('Epoch: {}, loss = {}'.format(i, loss))
        # torch.save(model.cpu().state_dict(), 'logs/model.pth')
        if torch.cuda.device_count() > 0:
            model.cuda()
    if dist.get_rank() == 0:
        acc = evaluate(model, feat_test, label_test)
        with open('logs/basis00/acc_test_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'.txt', 'w') as f:
            f.write(str(acc)+'\n')
    acc_train = evaluate(model, feat, label)
    with open('logs/basis00/acc_train_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank())+'.txt', 'w') as f:
        f.write(str(acc_train)+'\n')
    np.save('logs/basis00/loss_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank()), loss_list)

def parallel_train(rank, world_size, K, p, M):
    # load data
    #feat_file = 'data/wine.data'
    #feat, label = [], []
    #with open(feat_file, 'r') as f:
    #    for line in f.readlines():
    #        parts = line.strip().split(',')
    #        if float(parts[0]) > 2:
    #            continue
    #        label.append((float(parts[0])-1)*2-1)
    #        feat.append([float(part) for part in parts[1:]])
    #data_len = len(feat) // world_size + 1
    #feat = np.array(feat, dtype=np.float32)[rank*data_len:(rank+1)*data_len, np.newaxis, :]
    #label = np.array(label)[rank*data_len:(rank+1)*data_len]

    #data = np.loadtxt('data/iris_classes1and2_scaled.txt')
    #index = np.random.permutation(len(data))
    #data = data[index]
    #data_len = len(data) // world_size + (len(data) % world_size != 0)
    #label = data[:, -1][rank*data_len:(rank+1)*data_len]
    #feat = data[rank*data_len:(rank+1)*data_len, np.newaxis, :-1]
    #feat = np.concatenate((feat, feat), axis=-1)
    #print(feat.shape)

    feat_train = np.load('data/mnist_train_feat.npy')
    data_len = len(feat_train) // world_size + (len(feat_train) % world_size != 0)
    feat_train = feat_train[rank*data_len:(rank+1)*data_len]
    label_train = np.load('data/mnist_train_label.npy')
    label_train = label_train[rank*data_len:(rank+1)*data_len]
    feat_test = np.load('data/mnist_test_feat.npy')
    label_test = np.load('data/mnist_test_label.npy')

    # define model
    n_qubits = 6 #feat_train.shape[-1]
    n_layers = 4
    torch.manual_seed(0)
    model = QNN(n_qubits, n_layers, p=p, M=M, param_shift=True)
    # average_weights(model)

    # parallel wrapper
    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
        model.cuda()

    batch_size = 64 // world_size
    train(model, feat_train, label_train, feat_test, label_test, batch_size=batch_size, local_iter=K, p=p, M=M)

def init_process(rank, size, K, port, p, M, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.set_num_threads(1)
    fn(rank, size, K, p, M)

if __name__ == '__main__':
    opt = get_opt()
    # mp.set_start_method("spawn")
    # cost = []
    # for world_size in tqdm([opt.W], desc='world_size'):
    #     #world_size = 2#mp.cpu_count()

    #     processes = []
    #     st = time.time()
    #     for rank in range(world_size):
    #         p = mp.Process(target=init_process, args=(rank, world_size, opt.K, opt.port+29500, opt.p, opt.M, parallel_train))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()
    #     cost.append(time.time() - st)
    # np.save('logs/basis00/time'+str(opt.W)+'_'+str(opt.K)+'_'+str(opt.p)+'_'+str(opt.M), cost)
    parallel_train(0, 1, 1, 0.05, 100)
