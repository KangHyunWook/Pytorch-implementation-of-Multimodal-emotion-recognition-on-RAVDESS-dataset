import os
import pickle
import numpy as np
import argparse
from random import random

from torch import optim
from sklearn.metrics import f1_score   

import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.utils import shuffle
from models import AudioVisualModel

def eval(data, labels, mode=None, to_print=False):
    assert(mode is not None)

    model.eval()

    y_true, y_pred = [], []
    eval_loss, eval_loss_diff = [], []

    if mode == "test":
        if to_print:
            model.load_state_dict(torch.load(
                f'checkpoints/model_saved.pth'))
    
    corr=0
    with torch.no_grad():
        for i in range(0, len(data), 60):
            model.zero_grad()
            # v, a, y, l = batch
            d=data[i:i+60]
            l=labels[i:i+60]
            d=np.expand_dims(d,axis=0)
            v=torch.from_numpy(d[:, :, :35]).float()
            a=torch.from_numpy(d[:, :, 35:]).float()
            y=torch.from_numpy(l).float()

            v = to_gpu(v)
            a = to_gpu(a)
            y = to_gpu(y)

            y_tilde = model(v, a)
            
            cls_loss =  criterion(y_tilde, y)
            loss = cls_loss

            eval_loss.append(loss.item())
            preds=y_tilde.detach().cpu().numpy()
            y_trues=y.detach().cpu().numpy()
            
            for j in range(len(preds)):
                pred=np.argmax(preds[j])
                y_true=np.argmax(y_trues[j])
                if pred==y_true:
                    corr+=1        

    eval_loss = np.mean(eval_loss)
   
    accuracy = corr/(1.0*len(labels))

    return eval_loss, accuracy


def to_gpu(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
    return x

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

if __name__ == '__main__':
    data=[]
    labels=[]
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path')
    args=ap.parse_args()

    args=vars(args)
    
    with open(args['data_path'], 'rb') as out:
        au_mfcc = pickle.load(out)
    
    for key in au_mfcc:
        emotion = key.split('-')[2]
        emotion=int(emotion)-1
        labels.append(emotion)
        data.append(au_mfcc[key])
     
    data=np.array(data)
    labels=np.array(labels)
    
    labels=labels.reshape(labels.shape+(1,))
    data=np.hstack((data,labels))
    fdata=shuffle(data)
    data=fdata[:,:-1]
    labels = fdata[:,-1].astype(int)
    
    #todo: convert to one-hot
    
    new_labels= np.zeros((labels.shape[0], np.unique(labels.astype(int)).size))
    
    for i in range(len(labels)):
        new_labels[i, labels[i]]=1

    labels=new_labels

    test_data=data[-181:-1]
    test_labels=labels[-181:-1]
    data=data[:-180]
    labels=labels[:-180]
    
    train_data=data[:1020]
    train_labels=labels[:1020]
    
    dev_data=data[1020:]
    dev_labels=labels[1020:]
    
    # Setting random seed
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    

    is_train=True
    
    cuda=True
    
    model = AudioVisualModel()
    
    # Final list
    for name, param in model.named_parameters():
        
        if 'weight_hh' in name:
            nn.init.orthogonal_(param)
        print('\t' + name, param.requires_grad)
    
    model.cuda()

    optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4)

    patience = 6
    num_trials = 1
    
    criterion = nn.MSELoss(reduction="mean")

    # loss_diff = DiffLoss()
    loss_recon = MSE()
    loss_cmd = CMD()
    
    best_valid_loss = float('inf')
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    
    train_losses = []
    valid_losses = []
    
    curr_patience=6
   
    for e in range(50):
        model.train()

        train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
        train_loss_recon = []
        train_loss_sp = []
        train_loss = []
        #in train func
        for i in range(0, len(train_data), 60):
            
            data=train_data[i:i+60]
            label=train_labels[i:i+60]
       
            model.zero_grad()
            # v, a, y, l = batch
            data=np.expand_dims(data,axis=0)
            v=torch.from_numpy(data[:, :, :35]).float()
            a=torch.from_numpy(data[:, :, 35:]).float()
            y=torch.from_numpy(label).float()
            
            v = to_gpu(v)
            a = to_gpu(a)
            y = to_gpu(y)
            
            y_tilde = model(v, a)
            
            cls_loss = criterion(y_tilde, y)
            
            loss = cls_loss

            loss.backward()
            
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() 
                        if param.requires_grad], 1.0)
            optimizer.step()

            train_loss_cls.append(cls_loss.item())
            train_loss.append(loss.item())

        train_losses.append(train_loss)
     
        print(f"Training loss: {round(np.mean(train_loss), 4)}")
        
        valid_loss, valid_acc = eval(dev_data, dev_labels, mode="dev")
        print('valid_loss: {:.4f}, valid_acc: {:.4f}'.format(valid_loss, valid_acc))
        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")    
        
        
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/model_saved.pth')
            torch.save(optimizer.state_dict(), f'checkpoints/optim_saved.pth')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load(f'checkpoints/model_saved.pth'))
                optimizer.load_state_dict(torch.load(f'checkpoints/optim_saved.pth'))
                lr_scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break
        
    test_loss, test_acc=eval(test_data, test_labels, mode="test", to_print=True) 
    print('test_loss: {:.4f} test_acc: {:.4f}'.format(test_loss,test_acc))
    