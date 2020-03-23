import torch
import torch.nn as nn
import numpy as np
import pickle
import gc as g_c
import pdb
import warnings
import sys
import glob as gb
import os
import time
import random
from torch.nn import init
from tensorboardX import SummaryWriter
from math import sqrt
from config import *

#define the initial function to init the layer's parameters for the network
def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr_ = lr*0.995
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def to_tensor(x):
    if type(x) == list:
        x = np.array(x)
    return torch.tensor(torch.from_numpy(x), dtype=DTYPE, requires_grad=True)

def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f,encoding='iso-8859-1')

def save_obj(name,bag):
    with open (name,'wb') as f:
        pickle.dump(bag, f,protocol = 4)

def calculate_IoU(array_a, array_b):

    l_min = min(array_a[0], array_b[0])
    l_max = max(array_a[0], array_b[0])
    r_min = min(array_a[1], array_b[1])
    r_max = max(array_a[1], array_b[1])

    I = max(r_min - l_max + 1, 0)
    U = r_max - l_min + 1
    return I * 1.0 / U

def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL

class Loss_ali(nn.Module):
    def __init__(self, hyper_b):
        super().__init__()
        self.hyper_b = hyper_b

    def forward(self, score_mat,mask1,mask2):
        score_mat = score_mat * mask1
        loss = torch.log(1+torch.exp(score_mat))
        for i in range(batch_size):
            for j in range(batch_size):
                if torch.isinf(loss[i][j]):
                    loss[i][j] = score_mat[i][j]
        loss = mask2 * loss
        return torch.sum(loss) / batch_size

def nms_temporal(proposal,score, overlap):
    pick = []
    s= score.reshape(-1)
    x1 = proposal[:,0].reshape(-1)
    x2 = proposal[:,1].reshape(-1)
    union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index
    while len(I)>0:
        i = I[-1]
        pick.append(i)
        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

def do_eval(main_model,test_set)
    whole_top_5 = {0.6:0,0.1:0,0.7:0,0.2:0,0.8:0,0.3:0,0.9:0,0.4:0,0.95:0,0.5:0}
    whole_clip = 0
    top = 1
    for movie_name in test_set.movie_names:
        top_5 = {0.6:0,0.1:0,0.7:0,0.2:0,0.8:0,0.3:0,0.9:0,0.4:0,0.95:0,0.5:0}
        print("test %s"%movie_name)
        movie_clip_featmaps, movie_clip_sentences=test_set.load_movie_slidingclip(movie_name, 16)
        with torch.no_grad():
            fv = [movie_clip_featmaps[i][1] for i in range(len(movie_clip_featmaps))]
            fs = [movie_clip_sentences[i][1] for i in range(len(movie_clip_sentences))]

            annos = np.array([[int(movie_clip_sentences[i][0].split('_')[1]),int(movie_clip_sentences[i][0].split('_')[2])] for i in range(len(movie_clip_sentences))])
            proposals = [[int(movie_clip_featmaps[i][0].split('_')[1]),int(movie_clip_featmaps[i][0].split('_')[2])] for i in range(len(movie_clip_featmaps))]
            fv = torch.from_numpy(np.array(fv)).to(device)
            fs = torch.from_numpy(np.array(fs)).to(device)
            triple_score,_,_ = main_model(fv,fs)
            triple_score = triple_score.cpu()
            score_mat = triple_score[:,:,0]
            shift = triple_score[:,:,1:]
            shift = shift.cpu().numpy()
            re_score,ix = torch.sort(score_mat,dim=1,descending=True)
            ix = ix.cpu().numpy()
            shift_ = np.zeros((len(movie_clip_sentences),top,2))
            ix = ix[:,:top].reshape((-1))
            proposal = np.array([proposals[ix[i]] for i in range(ix.shape[0])]).reshape((len(movie_clip_sentences),top,2))
            for index in range(ix.shape[0]):
                i = int(index/top)
                j = int(index%top)
                k = ix[index]
                shift_[i][j] = shift[i][k]

            proposal = proposal.astype(np.float)
            proposal += shift_

            iou = np.zeros((len(movie_clip_sentences),top))
            for th in list(top_5.keys()):
                for i in range(iou.shape[0]):
                    for j in range(top):
                        iou[i][j] = calculate_IoU(annos[i],proposal[i][j])

                    if np.max(iou[i])>=th:
                        top_5[th] += 1
                        whole_top_5[th] += 1
            whole_clip += len(movie_clip_sentences)
        fv.cpu()
        fs.cpu()
        del fv,fs,triple_score,score_mat,shift,ix,re_score
        g_c.collect()
        torch.cuda.empty_cache()

        print(1.0*np.array(sorted(top_5.values(),reverse=True))/len(movie_clip_sentences))
    print(1.0*np.array(sorted(whole_top_5.values(),reverse=True))/whole_clip)
