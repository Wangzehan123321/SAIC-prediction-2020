import os
import pandas as pd
import numpy as np
from shapely.geometry import LinearRing, LineString, Point, Polygon
import pickle as pkl

#原始csv文件的根路径。
root_path="./data"
list_csv=os.listdir(root_path)

dict_list_num={}
train_data_all=np.empty((0,6))
#一共有246个csv文件,车辆最大索引为399,中心线最大索引为-66。
track_data_all=np.empty((246,399+1),dtype=object)
centerline_data_all=np.empty((246,66+1),dtype=object)

for num in range(len(list_csv)):
    # 将csv文件建立索引。
    dict_list_num[list_csv[num]]=num

    path_csv=os.path.join(root_path,list_csv[num])
    data=pd.read_csv(path_csv)
    array_data=np.array(data)

    #解决车道中心线重复记录的问题，即只保留第一次记录的车道中心线情况，去掉后续重复的。
    array_data_new=[]
    array_data_name=[]
    start=0
    for i in range(len(array_data)-1):
        if array_data[i,0]!=array_data[i+1,0]:
            if array_data[i,0] not in array_data_name:
                array_data_new.append(array_data[start:i+1])
                array_data_name.append(array_data[i,0])
                start=i+1
            else:
                start=i+1
    array_data=np.concatenate(array_data_new,0)

    #将车辆和车道中心线标签分类（这里不区分自车和障碍物，都参与训练）。
    number_list_all=set(array_data[:,0].tolist())
    number_vehicles=[]
    number_centerlines=[]
    for i in number_list_all:
        if i>=-1:
            number_vehicles.append(i)
        else:
            number_centerlines.append(i)
    train_data_csv = []
    for vehicle in number_vehicles:
        trajectory = array_data[np.argwhere(array_data[:, 0] == vehicle), :].squeeze(1)
        #为了解决原始文件中index大小与时间相反的问题。
        trajectory = np.flip(trajectory,0)

        if vehicle == -1:
            vehicle_id = 0
        else:
            vehicle_id = vehicle
        track_data_all[int(num)][int(vehicle_id)]=trajectory[:,[1,2,3]]

        if len(trajectory)>20:#选择历史序列2s来预测，所以可以使用的样本必须保证至少有2s的轨迹。
            trajectory_train=trajectory[20:-2,[1,2,3]]#x,y,timestamp。
            for i in range(trajectory_train.shape[0]):
                current_point=Point(trajectory_train[i,0:2])
                min_distance=100#初始化一个最小距离。
                for centerline in number_centerlines:
                    array_centerline = array_data[np.argwhere(array_data[:, 0] == centerline), :].squeeze(1)[:, [1, 2]]
                    if array_centerline.shape[0]<2:
                        continue
                    line = LineString(array_centerline)
                    if current_point.distance(line) < min_distance:
                        min_distance = current_point.distance(line)
                        oracle_ct = -centerline#将车道线标签变为正。
                if min_distance>=1.5:
                    pass#如果轨迹距离车道中心线的距离过远，那么就不用深度学习的方法来训练和预测。直接采用CV模型预测。
                else:
                    oracle_ct_array = array_data[np.argwhere(array_data[:, 0] == -oracle_ct), :].squeeze(1)[:, [1, 2]]
                    #处理车道中心线之间的组合和连续的问题。
                    success_lane_id=[-oracle_ct]
                    for centerline in number_centerlines:
                        if centerline not in success_lane_id:
                            array_centerline = array_data[np.argwhere(array_data[:, 0] == centerline), :].squeeze(1)[:,
                                           [1, 2]]
                            oracle_left=Point(oracle_ct_array[0,:])
                            oracle_right=Point(oracle_ct_array[-1,:])
                            point_left=Point(array_centerline[0,:])
                            point_right=Point(array_centerline[-1,:])
                            if oracle_left.distance(point_left)<=1:
                                oracle_ct_array=np.concatenate((np.flip(array_centerline,0),oracle_ct_array),0)
                                success_lane_id.append(centerline)
                            elif oracle_left.distance(point_right)<=1:
                                oracle_ct_array=np.concatenate((array_centerline,oracle_ct_array),0)
                                success_lane_id.append(centerline)
                            elif oracle_right.distance(point_left)<=1:
                                oracle_ct_array=np.concatenate((oracle_ct_array,array_centerline),0)
                                success_lane_id.append(centerline)
                            elif oracle_right.distance(point_right)<=1:
                                oracle_ct_array=np.concatenate((oracle_ct_array,np.flip(array_centerline,0)),0)
                                success_lane_id.append(centerline)

                    centerline_data_all[int(num)][int(oracle_ct)]=oracle_ct_array

                    #样本数据集的形式：[list_csv_index,vehicle_id,timestamp,x,y,centerline_id]。
                    train_data_csv.append([num,vehicle_id,trajectory_train[i,2],trajectory_train[i,0],trajectory_train[i,1],oracle_ct])
    train_data_all=np.concatenate((train_data_all,np.array(train_data_csv)),0)

save_path="./pkl_file/SAIC_sample.pkl"
data={}
data["train_data"]=train_data_all
data["track"]=track_data_all
data["centerline"]=centerline_data_all
with open(save_path,"wb") as f:
    pkl.dump(data,f)


#总样本按照8:2的比例划分训练集和验证集
import pickle as pkl
with open(save_path,"rb")as f:
    file=pkl.load(f)
data_all=file["train_data"]
track_all=file["track"]
center_all=file["centerline"]

sample_num=data_all.shape[0]
sample_list=list(range(sample_num))
train_list=np.random.choice(sample_num, int(0.8*sample_num), replace=False).tolist()
for num in train_list:
    sample_list.remove(num)
val_list=sample_list

train_all=data_all[train_list,:]
val_all=data_all[val_list,:]
train_dict={}
train_dict["train_data"]=train_all
train_dict["track"]=track_all
train_dict["centerline"]=center_all

val_dict={}
val_dict["train_data"]=val_all
val_dict["track"]=track_all
val_dict["centerline"]=center_all

train_save_path="./pkl_file/SAIC_train.pkl"
val_save_path="./pkl_file/SAIC_val.pkl"
with open(train_save_path,"wb")as f1:
    pkl.dump(train_dict,f1)
with open(val_save_path,"wb")as f2:
    pkl.dump(val_dict,f2)
