import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
root_path="/home/wangzehan/argoverse-forecasting/SAIC/data"
list_csv=os.listdir(root_path)
# print(list_csv[24])
path_csv=os.path.join(root_path,list_csv[62])
data=pd.read_csv(path_csv)
array_data=np.array(data)
number_list_all=set(array_data[:,0].tolist())
number_agents=[]
number_centerlines=[]
for i in number_list_all:
    if i>-1:
        number_agents.append(i)
    if i<-1:
        number_centerlines.append(i)
#agents
for agent in number_agents:
    array_agent=array_data[np.argwhere(array_data[:,0]==agent),:].squeeze(1)
    x_agent=array_agent[:,1]
    y_agent=array_agent[:,2]
    plt.plot(x_agent,y_agent,color="r",zorder=3)
    x_agent_final = x_agent[-1]
    y_agent_final = y_agent[-1]
    plt.scatter(x_agent_final, y_agent_final, color="chocolate", zorder=2)
#centerlines
print(number_centerlines)

# for centerline in number_centerlines:
for centerline in list(set(number_centerlines)):
    array_centerline=array_data[np.argwhere(array_data[:,0]==centerline),:].squeeze(1)#中心线可能会有重叠多次记录的问题
    ##将车辆中心线array进行sort
    array_centerline=array_centerline[np.argsort(array_centerline[:,0]),:]
    # if array_centerline.shape[0]==1:#去掉只有一个点的车道中心线
    #     continue()
    x_centerline=array_centerline[:,1]
    y_centerline=array_centerline[:,2]
    plt.plot(x_centerline,y_centerline,color="y",linewidth=1,linestyle="--",zorder=2)
    x_centerline_final = x_centerline[-1]
    y_centerline_final = y_centerline[-1]
    plt.scatter(x_centerline_final, y_centerline_final, color="chocolate",zorder=2)
#AV
array_AV=array_data[np.argwhere(array_data[:,0]==-1),:].squeeze(1)
if array_AV.shape[0]:
    x_AV=array_AV[:,1]
    y_AV=array_AV[:,2]
    plt.scatter(x_AV,y_AV,color="g")
    x_AV_final=x_AV[-1]
    y_AV_final=y_AV[-1]
    plt.scatter(x_AV_final,y_AV_final,color="chocolate")
plt.show()

# from shapely.geometry import LinearRing, LineString, Point, Polygon
# x=3
# y=5
# delta=0.01
# last=False
# point = Point(x, y)
# centerline=[[0,1],[0,2],[0,3],[0,4],[0,6]]
# centerline_ls = LineString(centerline)
# tang_dist = centerline_ls.project(point)#得到曲线上与其他点最近点的距离
# norm_dist = point.distance(centerline_ls)#得到垂向距离
# point_on_cl = centerline_ls.interpolate(tang_dist)#得到曲线上对应点的坐标
# # # Deal with last coordinate differently. Helped in dealing with floating point precision errors.
# if not last:
#     pt1 = point_on_cl.coords[0]
#     pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
#     pt3 = point.coords[0]
#
# else:
#     pt1 = centerline_ls.interpolate(tang_dist - delta).coords[0]
#     pt2 = point_on_cl.coords[0]
#     pt3 = point.coords[0]
#
# lr_coords = []
# lr_coords.extend([pt1, pt2, pt3])
# lr = LinearRing(lr_coords)
#
# # Left positive, right negative
# if lr.is_ccw:#如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
#     print(tang_dist, norm_dist)
# print(tang_dist, -norm_dist)