import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#原始csv文件的根路径
root_path="./data"
list_csv=os.listdir(root_path)
path_csv=os.path.join(root_path,list_csv[75])#选择第75个csv文件可视化
#print(list_csv[75])
data=pd.read_csv(path_csv)
array_data=np.array(data)

array_data_new=[]
array_data_name=[]

#为了处理车道中心线重复记录的问题，只保留第一次记录结果。
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
number_agents=[]
number_centerlines=[]
for i in number_list_all:
    if i>-1:
        number_agents.append(i)
    if i<-1:
        number_centerlines.append(i)

#agents visualization
for agent in number_agents:
    array_agent=array_data[np.argwhere(array_data[:,0]==agent),:].squeeze(1)
    x_agent=array_agent[:,1]
    y_agent=array_agent[:,2]
    plt.plot(x_agent,y_agent,color="r",zorder=3)
    x_agent_final = x_agent[-1]
    y_agent_final = y_agent[-1]
    plt.scatter(x_agent_final, y_agent_final, color="chocolate", zorder=2)
plt.plot(x_agent,y_agent,color="r",zorder=3,label="agent")

#centerlines visualization
for centerline in number_centerlines:
    array_centerline=array_data[np.argwhere(array_data[:,0]==centerline),:].squeeze(1)
    x_centerline=array_centerline[:,1]
    y_centerline=array_centerline[:,2]
    plt.plot(x_centerline,y_centerline,color="y",linewidth=1,linestyle="--",zorder=2)
    x_centerline_final = x_centerline[-1]
    y_centerline_final = y_centerline[-1]
    plt.scatter(x_centerline_final, y_centerline_final, color="chocolate",zorder=2)
plt.plot(x_centerline,y_centerline,color="y",linewidth=1,linestyle="--",zorder=2,label="centerline")

#AV visualization
array_AV=array_data[np.argwhere(array_data[:,0]==-1),:].squeeze(1)
if array_AV.shape[0]:
    x_AV=array_AV[:,1]
    y_AV=array_AV[:,2]
    plt.scatter(x_AV,y_AV,color="g",s=5,label="av")
    x_AV_final=x_AV[-1]
    y_AV_final=y_AV[-1]
    plt.scatter(x_AV_final,y_AV_final,color="chocolate")

plt.legend()
plt.show()
