from torch.utils.data import Dataset,DataLoader
import pandas as pd
import pickle as pkl
import numpy as np
import torch
from shapely.geometry import LinearRing, LineString, Point, Polygon
import random

## Dataset class for the NGSIM dataset
class SAICDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=1, enc_size = 64, data_argument=True):
        with open(mat_file,"rb")as f:
            raw_file=pkl.load(f)
        self.D = raw_file['train_data']
        self.T = raw_file['track']
        self.C = raw_file["centerline"]
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.data_len = len(self.D)
        self.data_argument=data_argument

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):#训练数据集的形式：list_csv_index,vehicle_id,timestamp,x,y,centerline_id
        while True:
            csv_Id = self.D[idx, 0].astype(int)
            veh_Id = self.D[idx, 1].astype(int)
            t = self.D[idx, 2]
            centerline=self.D[idx,5].astype(int)

            # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
            # hist,tang_dist_current = self.getHistory(csv_Id,veh_Id,centerline,t)
            # fut = self.getFuture(csv_Id,veh_Id,centerline,t,tang_dist_current)
            hist,current_point = self.getHistory(csv_Id, veh_Id, centerline, t)
            fut = self.getFuture(csv_Id,veh_Id,centerline,t,current_point)

            if self.data_argument:
                rightleft = random.random()
                if rightleft >= 0.5:
                    hist_out = np.full((hist.shape[0], hist.shape[1]), 0, dtype=float)
                    hist_out[:, 0] = hist[:, 0]
                    hist_out[:, 1] = -hist[:, 1]
                    fut_out = np.full((fut.shape[0], fut.shape[1]), 0, dtype=float)
                    fut_out[:, 0] = fut[:, 0]
                    fut_out[:, 1] = -fut[:, 1]
                else:
                    hist_out=hist
                    fut_out=fut
            else:
                hist_out = hist
                fut_out = fut
            if hist_out[0,0]>0:
                hist_out[:,0]=-hist_out[:,0]
                fut_out[:,0]=-fut_out[:,0]
            return hist_out,fut_out
            # if self.data_argument:#两个数据增强的方式（1、左右翻转。2、向上运动和向下运动）
            #     rightleft=random.random()
            #     updown=random.random()
            #     if rightleft>=0.5 and updown>=0.5:
            #         hist_out=np.full((hist.shape[0],hist.shape[1]),0,dtype=float)
            #         hist_out[:,0]=-hist[:,0]
            #         hist_out[:,1]=-hist[:,1]
            #         fut_out=np.full((fut.shape[0],fut.shape[1]),0,dtype=float)
            #         fut_out[:,0]=-fut[:,0]
            #         fut_out[:,1]=-fut[:,1]
            #     elif updown>=0.5:
            #         hist_out = np.full((hist.shape[0], hist.shape[1]), 0, dtype=float)
            #         hist_out[:, 0] = -hist[:, 0]
            #         hist_out[:, 1] = hist[:, 1]
            #         fut_out = np.full((fut.shape[0], fut.shape[1]), 0, dtype=float)
            #         fut_out[:, 0] = -fut[:, 0]
            #         fut_out[:, 1] = fut[:, 1]
            #     elif rightleft>=0.5:
            #         hist_out = np.full((hist.shape[0], hist.shape[1]), 0, dtype=float)
            #         hist_out[:, 0] = hist[:, 0]
            #         hist_out[:, 1] = -hist[:, 1]
            #         fut_out = np.full((fut.shape[0], fut.shape[1]), 0, dtype=float)
            #         fut_out[:, 0] = fut[:, 0]
            #         fut_out[:, 1] = -fut[:, 1]
            #     else:
            #         hist_out=hist
            #         fut_out=fut
            # else:
            #     hist_out = hist
            #     fut_out = fut
            # return hist_out,fut_out

    def getHistory(self, csv_Id, veh_Id, centerline, t):
        centerline=self.C[csv_Id,centerline]
        track=self.T[csv_Id,veh_Id]
        stpt = np.argwhere(track[:,2] == t).item()-self.t_h
        assert stpt>=0
        enpt = np.argwhere(track[:,2] == t).item()
        hist_track=track[stpt:enpt:self.d_s,0:2]
        current_point=hist_track[-1,:]
        return hist_track-current_point,current_point

    def getFuture(self, csv_Id, veh_Id, centerline, t,current_point):
        centerline = self.C[csv_Id, centerline]
        track = self.T[csv_Id, veh_Id]
        stpt = np.argwhere(track[:, 2] == t).item()+ self.d_s
        enpt =  np.minimum(len(track),np.argwhere(track[:, 2] == t).item() + self.t_f + 1)#这里在数据处理的时候还要设置当前时刻不能为序列最后时刻
        fut_track = track[stpt:enpt:self.d_s, 0:2]
        return fut_track-current_point

    # ## Helper function to get track history
    # def getHistory(self,csv_Id,veh_Id,centerline,t):
    #     centerline=self.C[csv_Id,centerline]
    #     track=self.T[csv_Id,veh_Id]
    #     stpt = np.argwhere(track[:,2] == t).item()-self.t_h
    #     assert stpt>=0
    #     enpt = np.argwhere(track[:,2] == t).item()
    #     hist_track=track[stpt:enpt:self.d_s,0:2]
    #     hist_track_rel=np.full((hist_track.shape[0],2),0,dtype=float)
    #     center_st=centerline[0,:]
    #     center_ed=centerline[-1,:]
    #     #需要填充centerline不够长的情况
    #     centerline_ls = LineString(centerline)
    #     delta=0.01
    #     for seq in range(hist_track.shape[0]):
    #         point=Point(hist_track[seq,:])
    #         tang_dist = centerline_ls.project(point)#得到曲线上与其他点最近点的距离
    #         norm_dist = point.distance(centerline_ls)#得到垂向距离(左侧为正，右侧为负)
    #         point_on_cl = centerline_ls.interpolate(tang_dist)#得到曲线上对应点的坐标
    #         #使用封闭曲线转向的顺时针或逆时针确定垂向坐标的正负
    #         pt1 = point_on_cl.coords[0]
    #         pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
    #         pt3 = point.coords[0]
    #         lr_coords = []
    #         lr_coords.extend([pt1, pt2, pt3])
    #         lr = LinearRing(lr_coords)
    #         if lr.is_ccw:#如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
    #             hist_track_rel[seq, 0] = tang_dist
    #             hist_track_rel[seq, 1] = norm_dist
    #         else:
    #             hist_track_rel[seq, 0] = tang_dist
    #             hist_track_rel[seq, 1] = -norm_dist
    #     tang_dist_current=hist_track_rel[-1,0]
    #     hist_track_rel[:,0]-=tang_dist_current
    #     return hist_track_rel,tang_dist_current
    #
    # ## Helper function to get track future
    # def getFuture(self,csv_Id,veh_Id,centerline,t,tang_dist_current):
    #     centerline = self.C[csv_Id, centerline]
    #     track = self.T[csv_Id, veh_Id]
    #     stpt = np.argwhere(track[:, 2] == t).item()+ self.d_s
    #     enpt =  np.minimum(len(track),np.argwhere(track[:, 2] == t).item() + self.t_f + 1)#这里在数据处理的时候还要设置当前时刻不能为序列最后时刻
    #     fut_track = track[stpt:enpt:self.d_s, 0:2]
    #     fut_track_rel = np.full((fut_track.shape[0], 2), 0,dtype=float)
    #     center_st = centerline[0, :]
    #     center_ed = centerline[-1, :]
    #     # 需要填充centerline不够长的情况
    #     centerline_ls = LineString(centerline)
    #     delta = 0.01
    #     for seq in range(fut_track.shape[0]):
    #         point = Point(fut_track[seq, :])
    #         tang_dist = centerline_ls.project(point)  # 得到曲线上与其他点最近点的距离
    #         norm_dist = point.distance(centerline_ls)  # 得到垂向距离
    #         point_on_cl = centerline_ls.interpolate(tang_dist)  # 得到曲线上对应点的坐标
    #         pt1 = point_on_cl.coords[0]
    #         pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
    #         pt3 = point.coords[0]
    #         lr_coords = []
    #         lr_coords.extend([pt1, pt2, pt3])
    #         lr = LinearRing(lr_coords)
    #         if lr.is_ccw:  # 如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
    #             fut_track_rel[seq, 0] = tang_dist
    #             fut_track_rel[seq, 1] = norm_dist
    #         else:
    #             fut_track_rel[seq, 0] = tang_dist
    #             fut_track_rel[seq, 1] = -norm_dist
    #     fut_track_rel[:, 0] -= tang_dist_current
    #     return fut_track_rel

    ## Collate function for dataloader
    def collate_fn(self, samples):
        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(self.t_h//self.d_s,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)

        for sampleId,(hist, fut) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
        return hist_batch, fut_batch, op_mask_batch


def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal

def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return lossVal, counts


# with open("/home/wangzehan/argoverse-forecasting/SAIC/train.pkl","rb")as f:
#     raw_file=pkl.load(f)
#
# csv_Id = raw_file["train_data"][1,0].astype(int)
# veh_Id = raw_file["train_data"][1,1].astype(int)
# t = raw_file["train_data"][1,2]
# centerline=raw_file["train_data"][1,5].astype(int)
# print(csv_Id,veh_Id,t,centerline)
# print(raw_file["train_data"][1,:])
# track=raw_file["track"][csv_Id,veh_Id]
# # track=track[np.argwhere(track[:, 2] == t).item(),:]
# # print(track)
# track=track[np.argwhere(track[:, 2] == t).item()-30:np.argwhere(track[:, 2] == t).item(),0:2]
# print(track)
# centerline=raw_file["centerline"][csv_Id,centerline]
# print(centerline)
# import matplotlib.pylab as plt
# print(np.max(centerline[:,0]))
# #centerline=centerline[[np.argmax(centerline[:,0]),np.argmin(centerline[:,0])],:]
# p1=centerline[np.argmax(centerline[:,0]),:]
# p2=centerline[np.argmax(centerline[:,1]),:]
# k=(p2[1]-p1[1])/(p2[0]-p1[0])
# b=p1[1]-k*p1[0]
# p1_new=np.array([0,0*k+b],dtype=float)
# p2_new=np.array([10000,10000*k+b],dtype=float)
# centerline=np.full((2,2),0,dtype=float)
# centerline[0]=p1_new
# centerline[1]=p2_new
# x_centerline = centerline[:, 0]
# y_centerline = centerline[:, 1]
# plt.plot(x_centerline, y_centerline, color="y", linewidth=1, linestyle="--", zorder=2)
# plt.show()
# hist_track_rel=np.full((track.shape[0],2),0,dtype=float)
# for i in range(track.shape[0]):
#     point=Point(track[i,:])
#     centerline_ls = LineString(centerline)
#     tang_dist = centerline_ls.project(point)#得到曲线上与其他点最近点的距离
#     norm_dist = point.distance(centerline_ls)#得到垂向距离(左侧为正，右侧为负)
#     # print(norm_dist)
#     point_on_cl = centerline_ls.interpolate(tang_dist)#得到曲线上对应点的坐标
#     point_on_cl.coords[0]#TODO:确认下point_on_cl的延伸机制
#     #使用封闭曲线转向的顺时针或逆时针确定垂向坐标的正负
#     delta=0.01
#     pt1 = point_on_cl.coords[0]
#     pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
#     pt3 = point.coords[0]
#     lr_coords = []
#     lr_coords.extend([pt1, pt2, pt3])
#     lr = LinearRing(lr_coords)
#     if lr.is_ccw:#如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
#         hist_track_rel[i, 0] = tang_dist
#         hist_track_rel[i, 1] = norm_dist
#     else:
#         hist_track_rel[i, 0] = tang_dist
#         hist_track_rel[i, 1] = -norm_dist
# # tang_dist_current=hist_track_rel[-1,0]
# # hist_track_rel[:,0]-=tang_dist_current
# print(hist_track_rel)


#data=SAICDataset("/home/wangzehan/argoverse-forecasting/SAIC/pkl_file/train_origin.pkl")
#print(data[100])
# dataloader=DataLoader(data,batch_size=10,shuffle=True,num_workers=8,collate_fn=data.collate_fn)
# hist,fut,op_mask=next(iter(dataloader))
# print(hist.shape)
# print(fut.shape)
# print(op_mask.shape)

# num=0
# for hist,fut in data:
#     if hist[0,0]<0:
#         num+=1
# print(num)