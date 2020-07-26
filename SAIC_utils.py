import math
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset,DataLoader
import pickle as pkl
import numpy as np
import torch
from shapely.geometry import LinearRing, LineString, Point, Polygon
import random

#将坐标从相对车道线坐标系转化到世界坐标系。
def get_xy_from_nt_seq(nt_seq: np.ndarray,
                       centerlines: List[np.ndarray]) -> np.ndarray:
    """Convert n-t coordinates to x-y, i.e., convert from centerline curvilinear coordinates to map coordinates.

    Args:
        nt_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension has 'n' (offset from centerline) and 't' (distance along centerline)
        centerlines (list of numpy array): Centerline for each track
    Returns:
        xy_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension contains coordinates in map frame

    """
    seq_len = nt_seq.shape[1]

    xy_seq = np.zeros(nt_seq.shape)
    for i in range(nt_seq.shape[0]):
        curr_cl = centerlines[i]
        line_string = LineString(curr_cl)
        for time in range(seq_len):

            offset_from_cl = nt_seq[i][time][0]
            dist_along_cl = nt_seq[i][time][1]
            x_coord, y_coord = get_xy_from_nt(offset_from_cl, dist_along_cl,
                                              curr_cl)
            xy_seq[i, time, 0] = x_coord
            xy_seq[i, time, 1] = y_coord

    return xy_seq


def get_xy_from_nt(n: float, t: float,
                   centerline: np.ndarray) -> Tuple[float, float]:
    """Convert a single n-t coordinate (centerline curvilinear coordinate) to absolute x-y.

    Args:
        n (float): Offset from centerline
        t (float): Distance along the centerline
        centerline (numpy array): Centerline coordinates
    Returns:
        x1 (float): x-coordinate in map frame
        y1 (float): y-coordinate in map frame

    """
    line_string = LineString(centerline)

    point_on_cl = line_string.interpolate(
        t) if t > 0 else line_string.interpolate(0)
    local_ls = None

    for i in range(len(centerline) - 1):
        pt1 = centerline[i]
        pt2 = centerline[i + 1]
        ls = LineString([pt1, pt2])
        if ls.distance(point_on_cl) < 1e-8:
            local_ls = ls
            break

    assert local_ls is not None, "XY from N({}) T({}) not computed correctly".format(
        n, t)

    pt1, pt2 = local_ls.coords[:]
    x0, y0 = point_on_cl.coords[0]

    if pt1[0] == pt2[0]:
        m = 0
        x1_1, x1_2 = x0 + n, x0 - n
        y1_1, y1_2 = y0, y0
    # Horizontal
    elif pt1[1] == pt2[1]:
        m = float("inf")
        x1_1, x1_2 = x0, x0
        y1_1, y1_2 = y0 + n, y0 - n
    # General case
    else:
        ls_slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        m = -1 / ls_slope

        x1_1 = x0 + n / math.sqrt(1 + m**2)
        y1_1 = y0 + m * (x1_1 - x0)
        x1_2 = x0 - n / math.sqrt(1 + m**2)
        y1_2 = y0 + m * (x1_2 - x0)

    # Rings formed by pt1, pt2 and coordinates computed above
    lr1 = LinearRing([pt1, pt2, (x1_1, y1_1)])
    lr2 = LinearRing([pt1, pt2, (x1_2, y1_2)])

    # If ring is counter clockwise
    if lr1.is_ccw:
        x_ccw, y_ccw = x1_1, y1_1
        x_cw, y_cw = x1_2, y1_2
    else:
        x_ccw, y_ccw = x1_2, y1_2
        x_cw, y_cw = x1_1, y1_1

    # If offset is positive, coordinate on the left
    if n > 0:
        x1, y1 = x_ccw, y_ccw
    # Else, coordinate on the right
    else:
        x1, y1 = x_cw, y_cw

    return x1, y1

## Dataset class for the SAIC dataset
class SAICDataset(Dataset):

    def __init__(self, mat_file, t_h=20, t_f=50, d_s=1, enc_size = 64, data_augment=False,val_metric=False,save_pkl=False):
        with open(mat_file,"rb")as f:
            raw_file=pkl.load(f)
        self.val_metric=val_metric
        self.save_pkl=save_pkl
        self.D = raw_file['train_data']
        self.T = raw_file['track']
        self.C = raw_file["centerline"]
        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.enc_size = enc_size
        self.data_len = len(self.D)
        self.data_augment=data_augment#是否进行数据增强

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):#数据集的形式：[list_csv_index,vehicle_id,timestamp,x,y,centerline_id]
        while True:
            csv_Id = self.D[idx, 0].astype(int)
            veh_Id = self.D[idx, 1].astype(int)
            t = self.D[idx, 2]
            centerline=self.D[idx,5].astype(int)

            hist,current_point = self.getHistory(csv_Id, veh_Id, centerline, t)
            fut = self.getFuture(csv_Id,veh_Id,centerline,t,current_point)

            if self.data_augment:
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

            if self.val_metric:
                oracle_ct = self.C[csv_Id, centerline]
                if self.save_pkl:
                    return hist_out,fut_out,current_point,oracle_ct,csv_Id,veh_Id
                else:
                    return hist_out,fut_out,current_point,oracle_ct
            else:
                return hist_out,fut_out

    ## Helper function to get track history
    def getHistory(self,csv_Id,veh_Id,centerline,t):
        centerline=self.C[csv_Id,centerline]
        track=self.T[csv_Id,veh_Id]
        stpt = np.argwhere(track[:,2] == t).item()-self.t_h+self.d_s
        assert stpt>=0
        enpt = np.argwhere(track[:,2] == t).item()+self.d_s
        hist_track=track[stpt:enpt:self.d_s,0:2]
        hist_track_rel=np.full((hist_track.shape[0],2),0,dtype=float)

        centerline_ls = LineString(centerline)
        delta=0.01
        for seq in range(hist_track.shape[0]):
            point=Point(hist_track[seq,:])
            tang_dist = centerline_ls.project(point)#得到曲线上与其他点最近点的距离
            norm_dist = point.distance(centerline_ls)#得到垂向距离(左侧为正，右侧为负)
            point_on_cl = centerline_ls.interpolate(tang_dist)#得到曲线上对应点的坐标
            #使用封闭曲线转向的顺时针或逆时针确定垂向坐标的正负
            pt1 = point_on_cl.coords[0]
            pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
            pt3 = point.coords[0]
            lr_coords = []
            lr_coords.extend([pt1, pt2, pt3])
            lr = LinearRing(lr_coords)
            if lr.is_ccw:#如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
                hist_track_rel[seq, 0] = norm_dist
                hist_track_rel[seq, 1] = tang_dist
            else:
                hist_track_rel[seq, 0] = -norm_dist
                hist_track_rel[seq, 1] = tang_dist
        tang_dist_current=hist_track_rel[-1,1]
        hist_track_rel[:,1]-=tang_dist_current
        return hist_track_rel,tang_dist_current

    ## Helper function to get track future
    def getFuture(self,csv_Id,veh_Id,centerline,t,tang_dist_current):
        centerline = self.C[csv_Id, centerline]
        track = self.T[csv_Id, veh_Id]
        stpt = np.argwhere(track[:, 2] == t).item()+ self.d_s
        enpt =  np.minimum(len(track),np.argwhere(track[:, 2] == t).item() + self.t_f + self.d_s)#这里在数据处理的时候还要设置当前时刻不能为序列最后时刻
        fut_track = track[stpt:enpt:self.d_s, 0:2]
        fut_track_rel = np.full((fut_track.shape[0], 2), 0,dtype=float)

        centerline_ls = LineString(centerline)
        delta = 0.01
        for seq in range(fut_track.shape[0]):
            point = Point(fut_track[seq, :])
            tang_dist = centerline_ls.project(point)  # 得到曲线上与其他点最近点的距离
            norm_dist = point.distance(centerline_ls)  # 得到垂向距离
            point_on_cl = centerline_ls.interpolate(tang_dist)  # 得到曲线上对应点的坐标
            pt1 = point_on_cl.coords[0]
            pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
            pt3 = point.coords[0]
            lr_coords = []
            lr_coords.extend([pt1, pt2, pt3])
            lr = LinearRing(lr_coords)
            if lr.is_ccw:  # 如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
                fut_track_rel[seq, 0] = norm_dist
                fut_track_rel[seq, 1] = tang_dist
            else:
                fut_track_rel[seq, 0] = -norm_dist
                fut_track_rel[seq, 1] = tang_dist
        fut_track_rel[:, 1] -= tang_dist_current
        if self.val_metric:
            return fut_track
        else:
            return fut_track_rel

    ## Collate function for dataloader
    def collate_fn(self, samples):

        hist_batch = torch.zeros(self.t_h//self.d_s,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)

        if self.val_metric:
            current_y_refer_batch = torch.zeros(len(samples), 1)
            centerline_batch=[]
            if self.save_pkl:
                csv_batch=[]
                vehicle_batch=[]
                for sampleId, (hist, fut, current_y_refer, oracle_ct, csv_Id, veh_Id) in enumerate(samples):
                    hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
                    hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
                    fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
                    fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
                    op_mask_batch[0:len(fut), sampleId, :] = 1
                    current_y_refer_batch[sampleId, :] = current_y_refer
                    centerline_batch.append(oracle_ct)
                    csv_batch.append(csv_Id)
                    vehicle_batch.append(veh_Id)
                return hist_batch,fut_batch,op_mask_batch,current_y_refer_batch,centerline_batch,csv_batch,vehicle_batch

            else:
                for sampleId, (hist, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
                    hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
                    fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
                    fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
                    op_mask_batch[0:len(fut), sampleId, :] = 1
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    centerline_batch.append(oracle_ct)
                return hist_batch,fut_batch,op_mask_batch,current_y_refer_batch,centerline_batch

        for sampleId,(hist, fut) in enumerate(samples):
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
        return hist_batch, fut_batch, op_mask_batch

#作为argoverse数据集预训练的损失函数
def MSE(y_pred, y_gt):
    acc = torch.zeros_like(y_gt)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    lossVal = torch.mean(acc[:,:,0])
    return lossVal

#作为SAIC数据集训练的损失函数
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

#作为SAIC数据集的评价指标
def maskedRMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return lossVal, counts

#作为argoverse数据集ADE的评价指标
def ADE(y_pred, y_gt):
    acc = torch.zeros_like(y_gt)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    lossVal = torch.mean(acc[:,:,0])
    return lossVal

#作为argoverse数据集FDE的评价指标
def FDE(y_pred, y_gt):
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    lossVal = torch.mean(out)
    return lossVal

#作为运动多模态的仲裁公式
def MSElastdistance(y_pred, y_gt):
    acc = torch.zeros_like(y_pred[0, :, 0])
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    acc = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    return acc

## Dataset class for the argoverse dataset
class ArgoverseDataset(Dataset):
    def __init__(self, pkl_file, t_h=20, t_f=30, d_s=1,social_input=False,val_metric=False,save_pkl=False):
        with open(pkl_file,"rb")as f:
            raw_file=pkl.load(f)
        self.social_input = social_input
        self.val_metric = val_metric
        self.save_pkl = save_pkl
        self.traj = raw_file["FEATURES"].values
        if self.val_metric:
            self.centerline = raw_file["ORACLE_CENTERLINE"].values
        if self.save_pkl:
            self.sequence = raw_file["SEQUENCE"].values
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        traj = self.traj[idx]
        #得到预测目标的历史轨迹和未来轨迹
        if self.social_input:
            hist=traj[0:self.t_h:self.d_s,[9,10,6,7,8]]
        else:
            hist=traj[0:self.t_h:self.d_s,-2:]
        fut = traj[self.t_h::self.d_s,-2:]
        #将沿着车道线的坐标变化为相对坐标
        current_y_refer = hist[-1,1]
        hist[:,1]=hist[:,1]-current_y_refer
        fut[:,1]=fut[:,1]-current_y_refer

        if self.val_metric:
            oracle_ct=self.centerline[idx]
            fut = traj[self.t_h::self.d_s,[3,4]]
            if self.save_pkl:
                seq=self.sequence[idx]
                return seq, hist, fut, current_y_refer, oracle_ct
            else:
                return hist, fut, current_y_refer, oracle_ct
        else:
            return hist, fut


    def collate_fn(self, samples):
        #将samples数据整合到batch维度上（sequence,batch,2）
        if self.social_input:
            hist_traj_batch = torch.zeros(self.t_h, len(samples), 5)
        else:
            hist_traj_batch = torch.zeros(self.t_h, len(samples), 2)
        fut_batch = torch.zeros(self.t_f, len(samples), 2)
        current_y_refer_batch = torch.zeros(len(samples), 1)

        if self.val_metric:
            if self.save_pkl:
                seq_list=[]
                centerline_batch=[]
                for sampleId, (seq, hist_traj_all, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    seq_list.append(seq)
                    centerline_batch.append(oracle_ct)
                return hist_traj_batch,fut_batch,current_y_refer_batch,seq_list,centerline_batch
            else:
                centerline_batch=[]
                for sampleId, (hist_traj_all, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    centerline_batch.append(oracle_ct)
                return hist_traj_batch,fut_batch,current_y_refer_batch,centerline_batch
        else:
            for sampleId, (hist_traj_all, fut) in enumerate(samples):
                hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
            return hist_traj_batch,fut_batch