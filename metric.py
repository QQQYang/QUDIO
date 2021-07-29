from pennylane import numpy as np
import torch

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

def cross_entropy_loss(predict, label):
    '''
    Multi-class classifier
    '''
    p = softmax(predict)
    m = label.shape[0]
    log_likelihood = -np.log(p[range(m), label])
    return np.sum(log_likelihood) / m

def mse_loss(predict, label):
    return np.mean((predict - label)**2)

def cross_entropy(predict, label):
    '''
    Binary classifier
    '''
    prob_pos = np.exp(predict) / (np.exp(predict) + np.exp(1 - predict))
    prob_neg = 1 - prob_pos
    prob = np.concatenate((prob_pos[:, np.newaxis], prob_neg[:, np.newaxis]), axis=-1)
    label_onehot = ((label + 1) / 2)[:, np.newaxis]
    label_onehot = np.concatenate((label_onehot, 1 - label_onehot), axis=-1)
    prob = np.sum(prob * label_onehot, axis=-1)
    return np.mean(-prob * np.log(prob))

def accuracy(predicts, labels):
    '''
    Binary classifier
    '''
    assert len(predicts) == len(labels)
    return np.sum((np.sign(predicts)*labels+1)/2)/len(predicts)

def cost(param, circuit, feat, label, weight_decay=0, loss_name='MSE'):
    exp = [circuit(param, feat=feat[i]) for i in range(len(feat))]
    if loss_name == 'MSE':
        return mse_loss(np.array(exp), label) + weight_decay * np.linalg.norm(param)
    else:
        return cross_entropy_loss(np.array(exp), label) + weight_decay * np.linalg.norm(param)
