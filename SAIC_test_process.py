import os
import pandas as pd
import numpy as np
from shapely.geometry import LinearRing, LineString, Point, Polygon
import pickle as pkl

root_path="./data"
list_csv=os.listdir(root_path)

### 将csv文件建立索引。
dict_list_num={}
net_data_all=np.empty((0,6))#基于深度学习
phy_data_all=np.empty((0,6))#基于物理模型

#一共有246个csv文件,车辆最大索引为399,中心线最大索引为-66。
track_data_all=np.empty((246,399+1),dtype=object)
centerline_data_all=np.empty((246,66+1),dtype=object)

for num in range(len(list_csv)):

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

    number_list_all=set(array_data[:,0].tolist())
    #将车辆和车道中心线标签分类。
    number_vehicles=[]
    number_centerlines=[]
    for i in number_list_all:
        if i>-1:
            number_vehicles.append(i)
        if i<-1:
            number_centerlines.append(i)
    net_data_csv = []
    phy_data_csv = []
    pass_num=0
    for vehicle in number_vehicles:
        trajectory = array_data[np.argwhere(array_data[:, 0] == vehicle), :].squeeze(1)
        #原始文件中index记录与时间相反
        trajectory = np.flip(trajectory,0)

        vehicle_id = vehicle
        track_data_all[int(num)][int(vehicle_id)]=trajectory[:,[1,2,3]]

        trajectory_train=trajectory[:,[1,2,3]]#x,y,timestamp

        current_point=Point(trajectory_train[-1,0:2])

        min_distance=100#初始化一个最小距离
        for centerline in number_centerlines:
            array_centerline = array_data[np.argwhere(array_data[:, 0] == centerline), :].squeeze(1)[:, [1, 2]]
            if array_centerline.shape[0]<2:
                continue
            line = LineString(array_centerline)
            if current_point.distance(line) < min_distance:
                min_distance = current_point.distance(line)
                oracle_ct = -centerline#将车道线标签变为正
        if min_distance>=1.5:
            # 数据集的形式：[list_csv_index,vehicle_id,timestamp,x,y,centerline_id]
            phy_data_csv.append([num, vehicle_id, trajectory_train[-1, 2], trajectory_train[-1, 0], trajectory_train[-1, 1],oracle_ct])
        else:
            if len(trajectory_train)<20:
                pass_num+=1
                pass
            else:
                oracle_ct_array = array_data[np.argwhere(array_data[:, 0] == -oracle_ct), :].squeeze(1)[:, [1, 2]]
                # #新加的内容，处理车道中心线连续的问题
                # success_lane_id=[-oracle_ct]
                # for centerline in number_centerlines:
                #     if centerline not in success_lane_id:
                #         array_centerline = array_data[np.argwhere(array_data[:, 0] == centerline), :].squeeze(1)[:,
                #                                [1, 2]]
                #         oracle_left=Point(oracle_ct_array[0,:])
                #         oracle_right=Point(oracle_ct_array[-1,:])
                #         point_left=Point(array_centerline[0,:])
                #         point_right=Point(array_centerline[-1,:])
                #         if oracle_left.distance(point_left)<=1:
                #             oracle_ct_array=np.concatenate((np.flip(array_centerline,0),oracle_ct_array),0)
                #             success_lane_id.append(centerline)
                #         elif oracle_left.distance(point_right)<=1:
                #             oracle_ct_array=np.concatenate((array_centerline,oracle_ct_array),0)
                #             success_lane_id.append(centerline)
                #         elif oracle_right.distance(point_left)<=1:
                #             oracle_ct_array=np.concatenate((oracle_ct_array,array_centerline),0)
                #             success_lane_id.append(centerline)
                #         elif oracle_right.distance(point_right)<=1:
                #             oracle_ct_array=np.concatenate((oracle_ct_array,np.flip(array_centerline,0)),0)
                #             success_lane_id.append(centerline)

                centerline_data_all[int(num)][int(oracle_ct)]=oracle_ct_array
                 #数据集的形式：[list_csv_index,vehicle_id,timestamp,x,y,centerline_id]
                net_data_csv.append([num,vehicle_id,trajectory_train[-1,2],trajectory_train[-1,0],trajectory_train[-1,1],oracle_ct])

    if np.array(net_data_csv).shape[0]==0:
        pass
    else:
        net_data_all=np.concatenate((net_data_all,np.array(net_data_csv)),0)

    if np.array(phy_data_csv).shape[0]==0:
        pass
    else:
        phy_data_all=np.concatenate((phy_data_all,np.array(phy_data_csv)),0)
    assert pass_num+np.array(phy_data_csv).shape[0]+np.array(net_data_csv).shape[0]==len(number_vehicles)

net_save_path="test_file/net_data.pkl"
net_data_dict={}
net_data_dict["train_data"]=net_data_all
net_data_dict["track"]=track_data_all
net_data_dict["centerline"]=centerline_data_all
with open(net_save_path,"wb") as f1:
    pkl.dump(net_data_dict,f1)

phy_save_path="test_file/phy_data.pkl"
phy_data_dict={}
phy_data_dict["train_data"]=phy_data_all
phy_data_dict["track"]=track_data_all
with open(phy_save_path,"wb") as f2:
    pkl.dump(phy_data_dict,f2)

csv_save_path="test_file/csv_list.pkl"
with open(csv_save_path,"wb") as f3:
    pkl.dump(dict_list_num,f3)


