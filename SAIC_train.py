import torch
import numpy as np
from SAIC_model import SAICNet,init_weights
from SAIC_utils import SAICDataset,maskedMSE
from torch.utils.data import DataLoader
import time
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
import visdom
class Visualizer(object):
    def __init__(self,env='default',**kwargs):
        self.vis=visdom.Visdom(env=env,**kwargs)
        self.index={}
    def plot(self,name,y,**kwargs):
        x=self.index.get(name,0)
        self.vis.line(X=np.array([x]),Y=np.array([y]),win=name,opts=dict(title=name),update=None if x==0 else 'append',**kwargs)
        self.index[name]=x+1
    def txt(self,name,text,**kwargs):
        x=self.index.get(name,0)
        self.vis.text(text,win=name,append=False if x==0 else True,**kwargs)
        self.index[name]=x+1
vis=Visualizer('SAIC_on_argoverse_pretrained_net')

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 64
args['in_length'] = 20
args['out_length'] = 50
args['input_embedding_size'] = 32
args['pool_embedding_size'] = 64
args['num_lon_classes'] = 3
args['use_maneuvers'] = False
#新加的是否使用交互模块
args["social_input"] = False
args['interaction'] = False

# Initialize network
net = SAICNet(args)
#加载argoverse dataset的预训练模型
net.load_state_dict(torch.load('./trained_models/argo_lstm_pretrain/pretrained_net_28.tar'))
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
trainEpochs = 1000
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
batch_size = 128

## Initialize data loaders
trSet = SAICDataset('./pkl_files/SAIC_train.pkl')#train datatset
valSet = SAICDataset('./pkl_files/SAIC_val.pkl')#val dataset
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

## Variables holding train and validation loss values:
train_loss = []
val_loss = []

for epoch_num in range(trainEpochs):

    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0

    for i, data in enumerate(trDataloader):
        st_time = time.time()
        hist,fut,op_mask = data

        if args['use_cuda']:
            hist = hist.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        # Forward pass
        if args['use_maneuvers']:
            pass
            #TODO:概率的多模态轨迹预测
        else:
            fut_pred = net(hist)
            l = maskedMSE(fut_pred, fut, op_mask)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        #每100次记录1次训练的损失
        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'), "| ETA(s):",int(eta))
            text="Epoch no:"+str(epoch_num+1)+"| Epoch progress(%):"+str(i/(len(trSet)/batch_size)*100)+"| Avg train loss:"+str(avg_tr_loss/100)+"| ETA(s):"+str(int(eta))
            vis.txt(name="record",text=text)
            vis.plot(name="loss",y=int(avg_tr_loss/100))

            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    # _     ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    with torch.no_grad():
        print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
        avg_val_loss = 0
        val_batch_count = 0
        total_points = 0

        for j, data  in enumerate(valDataloader):
            st_time = time.time()
            hist, fut, op_mask = data

            if args['use_cuda']:
                hist = hist.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            # Forward pass
            if args['use_maneuvers']:
                pass
                #TODO:概率的多模态轨迹预测
            else:
                fut_pred = net(hist)
                l = maskedMSE(fut_pred, fut, op_mask)

            avg_val_loss += l.item()
            val_batch_count += 1

        print(avg_val_loss/val_batch_count)

        # Print validation loss and update display variables
        print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'))
        textnew='Validation loss :'+str(avg_val_loss/val_batch_count)
        vis.txt(name="val",text=textnew)
        vis.plot(name="loss_val",y=int(avg_val_loss/val_batch_count))
    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    torch.save(net.state_dict(), './trained_models/SAIC_on_argoverse_pretrained_net/cslstm_{}.tar'.format(str(epoch_num)))
