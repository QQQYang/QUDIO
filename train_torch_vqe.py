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
import json

from circuit.vqe_torch_pure import VQE
from circuit.vqe_torch_pure_he import VQE_HE

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--W", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mol", type=str, default='LiH')
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

def median_weights(model):
    size = dist.get_world_size()
    for param in model.parameters():
        param_gather = [torch.ones_like(param.data) for _ in range(size)]
        dist.all_gather(param_gather, param.data)
        param_median = torch.median(torch.stack(param_gather, dim=0), 0)[0]
        param.data = param_median

def random_weights(model):
    size = dist.get_world_size()
    for param in model.parameters():
         param_gather = [torch.ones_like(param.data) for _ in range(size)]
         dist.all_gather(param_gather, param.data)
         index = np.random.randint(0, size)
         param_median = torch.stack(param_gather, dim=0)[index]
         param.data = param_median

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

def train(model, local_iter=8, p=0.1, M=100, distance=0.3, seed=0, h={}, world_size=1, mol='LiH'):
    model.train()

    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': 0.4, 'weight_decay': 0}], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.5, last_epoch=-1)

    loss_list = []
    rank = dist.get_rank()
    for i in range(200):
        #np.random.shuffle(h)
        #data_len = len(h) // world_size + (len(h) % world_size != 0)
        #h_new = h[rank*data_len:(rank+1)*data_len]
        #h_part = {}
        #for j in range(len(h_new)):
        #    h_part[h_new[j][0]] = h_new[j][1]
        #model.hamiltonian = h_part

        loss = model()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % local_iter == 0:
            average_weights(model)
            #median_weights(model)
        scheduler.step()
        loss_list.append(loss.item())

        if dist.get_rank() == 0:
            print('Epoch: {}, loss = {}'.format(i, loss))
        # torch.save(model.cpu().state_dict(), 'logs/model.pth')
        if torch.cuda.device_count() > 0:
            model.cuda()
    average_weights(model)
    #median_weights(model)
    log_dir = 'logs/vqe/ideal/'+mol+'/baseline/'+str(seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
       	h_part = {}
        for j in range(len(h)):
        	h_part[h[j][0]] = h[j][1]
        model.hamiltonian = h_part
        loss = model()
        np.save(os.path.join(log_dir, 'energy_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank())+'_'+str(distance)), loss.item())
    np.save(os.path.join(log_dir, 'loss_'+str(dist.get_world_size())+'_'+str(local_iter)+'_'+str(p)+'_'+str(M)+'_'+str(dist.get_rank())+'_'+str(distance)), loss_list)

def parallel_train(rank, world_size, K, p, M, distance=0.3, seed=0, mol='LiH'):
    h = ''
    with open('VQE/data/'+mol+str(distance)+'.json', 'r') as f:
        h = json.load(f)
    h = list(h.items())
    # temporal for test
    np.random.seed(seed)
    #np.random.shuffle(h)
    data_len = len(h) // world_size + (len(h) % world_size != 0)
    h_new = h[rank*data_len:(rank+1)*data_len]
    h_part = {}
    for i in range(len(h_new)):
        h_part[h_new[i][0]] = h_new[i][1]

    # define model
    n_qubits = len(h_new[0][0]) #feat_train.shape[-1]
    n_layers = 4
    torch.manual_seed(seed)
    if mol == 'h2':
        model = VQE(n_qubits, h_part, p=p, M=M, param_shift=False)
    else:
        model = VQE_HE(n_qubits, h_part, p=p, M=M, param_shift=False)
    average_weights(model)

    # parallel wrapper
    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
        model.cuda()

    train(model, local_iter=K, p=p, M=M, distance=distance, seed=seed, h=h, world_size=world_size, mol=mol)

def init_process(rank, size, K, port, p, M, distance, seed, mol, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.set_num_threads(1)
    fn(rank, size, K, p, M, distance, seed=seed, mol=mol)

if __name__ == '__main__':
    opt = get_opt()
    mp.set_start_method("spawn")
    cost = []
    for distance in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]:
        for world_size in tqdm([opt.W], desc='world_size'):
            #world_size = 2#mp.cpu_count()

            processes = []
            st = time.time()
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size, opt.K, opt.port+29500, opt.p, opt.M, distance, opt.seed, opt.mol, parallel_train))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cost.append(time.time() - st)
    np.save('logs/vqe/ideal/'+opt.mol+'/baseline/'+str(opt.seed)+'/time'+str(opt.W)+'_'+str(opt.K)+'_'+str(opt.p)+'_'+str(opt.M), cost)

