import os
import pandas as pd
import numpy as np
from shapely.geometry import LinearRing, LineString, Point, Polygon
import pickle as pkl

# root_path="/home/wangzehan/argoverse-forecasting/SAIC/data"
# list_csv=os.listdir(root_path)
# number_vehicles = []
# number_centerlines = []
# #train_data_all=np.empty((0,6))
# for num in range(len(list_csv)):
#     path_csv = os.path.join(root_path, list_csv[num])
#     data=pd.read_csv(path_csv)
#     array_data = np.array(data)
#     number_list_all = set(array_data[:, 0].tolist())
#     for i in number_list_all:
#         if i > -1:
#             if i==0:
#                 print(i)
#             number_vehicles.append(i)
#         if i < -1:
#             number_centerlines.append(-i)
# print(max(number_vehicles))
# print(max(number_centerlines))

save_path="/home/wangzehan/argoverse-forecasting/SAIC/pkl_file/train_origin.pkl"
root_path="/home/wangzehan/argoverse-forecasting/SAIC/data"
list_csv=os.listdir(root_path)
#将csv文件建立索引
dict_list_num={}
train_data_all=np.empty((0,6))
track_data_all=np.empty((246,399+1),dtype=object)
centerline_data_all=np.empty((246,66+1),dtype=object)

#找出最大横坐标和最小横坐标的内容
# x_max=0(11000,-28000)
# x_min=10000

for num in range(len(list_csv)):
#for num in range(1):
    dict_list_num[list_csv[num]]=num

    path_csv=os.path.join(root_path,list_csv[num])
    data=pd.read_csv(path_csv)
    array_data=np.array(data)
    number_list_all=set(array_data[:,0].tolist())
    #将车辆和车道中心线标签分类（这里不区分自车和障碍物，都参与训练）
    number_vehicles=[]
    number_centerlines=[]
    number_centerlines_both=[]#用以计数重复的中心线
    for i in number_list_all:
        if i>=-1:
            number_vehicles.append(i)
        else:
            if i in number_centerlines:
                number_centerlines_both.append(i)
            number_centerlines.append(i)
    train_data_csv = []
    for vehicle in number_vehicles:
        trajectory = array_data[np.argwhere(array_data[:, 0] == vehicle), :].squeeze(1)
        if vehicle == -1:
            vehicle_id = 0
        else:
            vehicle_id = vehicle
        track_data_all[int(num)][int(vehicle_id)]=trajectory[:,[1,2,3]]

        # track_max=np.max(trajectory[:,1])
        # if track_max>x_max:
        #     x_max=track_max
        # track_min = np.min(trajectory[:, 1])
        # if track_min < x_min:
        #     x_min = track_min

        if len(trajectory)>30:#这里选择历史序列3s用来预测
            trajectory_train=trajectory[30:-2,[1,2,3]]#x,y,timestamp
            for i in range(trajectory_train.shape[0]):
                current_point=Point(trajectory_train[i,0:2])
                min_distance=100#初始化一个最小距离
                for j in number_centerlines:
                    array_centerline = array_data[np.argwhere(array_data[:, 0] == j), :].squeeze(1)[:, [1, 2]]
                    if array_centerline.shape[0]<2:
                        continue
                    line = LineString(array_centerline)
                    if current_point.distance(line) < min_distance:
                        min_distance = current_point.distance(line)
                        oracle_ct = -j#将车道线标签变为正
                if min_distance>=1.5:
                    pass#如果轨迹距离车道中心线的距离过远，那么就不用深度学习的方法来训练和预测
                else:
                    oracle_ct_array=array_data[np.argwhere(array_data[:, 0] == -oracle_ct), :].squeeze(1)[:, [1, 2]]

                    # p1 = oracle_ct_array[np.argmax(oracle_ct_array[:, 0]), :]
                    # p2 = oracle_ct_array[np.argmin(oracle_ct_array[:, 0]), :]
                    # k = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    # b = p1[1] - k * p1[0]
                    # x_min=-28000
                    # x_max=11000
                    # p1_new = np.array([x_min, x_min * k + b], dtype=float)
                    # p2_new = np.array([x_max, x_max * k + b], dtype=float)
                    # oracle_ct_array_new = np.full((2, 2), 0, dtype=float)
                    # oracle_ct_array_new[0] = p1_new
                    # oracle_ct_array_new[1] = p2_new
                    # centerline_data_all[int(num)][int(oracle_ct)]=oracle_ct_array_new
                    centerline_data_all[int(num)][int(oracle_ct)]=array_data[np.argwhere(array_data[:, 0] == -oracle_ct), :].squeeze(1)[:, [1, 2]]
                    # if vehicle==-1:
                    #     vehicle_id=0
                    # else:
                    #     vehicle_id=vehicle
                    #训练数据集的形式：list_csv_index,vehicle_id,timestamp,x,y,centerline_id
                    train_data_csv.append([num,vehicle_id,trajectory_train[i,2],trajectory_train[i,0],trajectory_train[i,1],oracle_ct])
    train_data_all=np.concatenate((train_data_all,np.array(train_data_csv)),0)

data={}
data["train_data"]=train_data_all
data["track"]=track_data_all
data["centerline"]=centerline_data_all
with open(save_path,"wb") as f:
    pkl.dump(data,f)

# print(x_max,x_min)
# import pickle as pkl
# with open("/home/wangzehan/argoverse-forecasting/SAIC/pkl_file/train_origin.pkl","rb")as f:
#     file=pkl.load(f)
# data_all=file["train_data"]
# track_all=file["track"]
# center_all=file["centerline"]
#
# print(data_all.shape)
#
# data_list=list(range(41294))
# train_list=np.random.choice(41294, 28906, replace=False).tolist()
# for num in train_list:
#     data_list.remove(num)
# assert len(data_list)==12388
#
# train_all=data_all[train_list,:]
#
# val_all=data_all[data_list,:]
# train_dict={}
# train_dict["train_data"]=train_all
# train_dict["track"]=track_all
# train_dict["centerline"]=center_all
#
# val_dict={}
# val_dict["train_data"]=val_all
# val_dict["track"]=track_all
# val_dict["centerline"]=center_all
# with open("/home/wangzehan/argoverse-forecasting/SAIC/pkl_file/train_origin_1.pkl","wb")as f1:
#     pkl.dump(train_dict,f1)
# with open("/home/wangzehan/argoverse-forecasting/SAIC/pkl_file/val_origin_1.pkl","wb")as f2:
#     pkl.dump(val_dict,f2)





# import os
# import pandas as pd
# import numpy as np
# from shapely.geometry import LinearRing, LineString, Point, Polygon
# import pickle as pkl
# root_path="/home/wangzehan/argoverse-forecasting/SAIC/data"
# list_csv=os.listdir(root_path)
# print(len(list_csv))
# number_vehicles = []
# number_centerlines = []
# #train_data_all=np.empty((0,6))
# for num in range(len(list_csv)):
#     path_csv = os.path.join(root_path, list_csv[num])
#     data=pd.read_csv(path_csv)
#     array_data = np.array(data)
#     number_list_all = set(array_data[:, 0].tolist())
#     for i in number_list_all:
#         if i > -1:
#             if i==0:
#                 print(i)
#             number_vehicles.append(i)
#         if i < -1:
#             number_centerlines.append(-i)
# print(max(number_vehicles))
# print(max(number_centerlines))
# a=np.empty((10,10),dtype=object)
# a[2][4]=np.array([1,2,3])
# print(a[2][4])



import pandas as pd
import numpy as np
import os
from shapely.geometry import LinearRing, LineString, Point, Polygon
root_path="/home/wangzehan/argoverse-forecasting/SAIC/data"
list_csv=os.listdir(root_path)
path_csv=os.path.join(root_path,list_csv[26])
data=pd.read_csv(path_csv)
array_data=np.array(data)
number_list_all=set(array_data[:,0].tolist())
trajectory=array_data[np.argwhere(array_data[:, 0] == 14), :].squeeze(1)
point=Point(trajectory[:,[1,2]][1,:])
print(point.coords[0])
# print(number_list_all)
# min_distance=100
# for i in number_list_all:
#     if i<-1:
#         array_centerline = array_data[np.argwhere(array_data[:, 0] == i), :].squeeze(1)[:,[1,2]]
#         line=LineString(array_centerline)
#         if point.distance(line)<min_distance:
#             min_distance=point.distance(line)
#             oracle_ct=array_centerline
#
# import matplotlib.pyplot as plt
# x_agent = trajectory[:, 1]#-trajectory[0,1]
# y_agent = trajectory[:, 2]#-trajectory[0,2]
# plt.plot(x_agent, y_agent, color="r", zorder=3)
# x_agent_final = x_agent[-1]
# y_agent_final = y_agent[-1]
# plt.scatter(x_agent_final, y_agent_final, color="chocolate", zorder=2)
#
# x_centerline = oracle_ct[:, 0]
# y_centerline = oracle_ct[:, 1]
# plt.plot(x_centerline, y_centerline, color="y", linewidth=1, linestyle="--", zorder=2)
#
# print(min_distance)
# plt.show()
