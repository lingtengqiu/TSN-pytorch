import numpy as np
from collections import OrderedDict
import torch
import fire
from sklearn.metrics import confusion_matrix

def vote(**kwargs):
    dict =[]
    if kwargs['Flow'] == True:
        dict.append(torch.load("result/Flow.pth"))
    if kwargs['RGB'] == True:
        dict.append(torch.load("result/RGB.pth"))
    if kwargs['RGBDiff'] == True:
        dict.append(torch.load("result/RGBDiff.pth"))
    labels = []
    preds = []
    for key in dict[0].keys():
        temp = []
        for i in range(len(dict)):
            temp.append(dict[i][key]['pred'])
            label = dict[i][key]['label']
        pred = np.argmax(np.concatenate(temp,axis=0).sum(0),0)
        preds.append(pred)
        labels.append(label)
    cfs  = confusion_matrix(labels,preds)
    cfs_num = cfs.sum(1)
    cfs_hit = np.diag(cfs)
    pred_prob = (cfs_hit*1./cfs_num)
    print np.mean(pred_prob)
if __name__ == '__main__':
    fire.Fire()