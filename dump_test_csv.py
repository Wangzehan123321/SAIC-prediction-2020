import pandas as pd
import pickle as pkl
import numpy as np

#实现基于物理模型CV的轨迹预测
root_path="test_file/phy_data.pkl"
with open(root_path,"rb") as f:
    file=pkl.load(f)
traj=file["train_data"]
track=file["track"]
data={}
for i in range(len(traj)):
    current_track=track[int(traj[i,0])][int(traj[i,1])]
    if len(current_track)<10:
        continue
    assert current_track[-1][2]==0
    x_0=current_track[-1][0]
    y_0=current_track[-1][1]
    x_1=current_track[-2][0]
    y_1=current_track[-2][1]
    x_2=current_track[-3][0]
    y_2=current_track[-3][1]
    vx=(x_0-x_2)/(2*0.1)
    vy=(y_0-y_2)/(2*0.1)
    predict_array=np.full((50,2),0,dtype=float)
    for j in range(50):
        predict_array[j][0]=vx*(j+1)*0.1+x_0
        predict_array[j][1]=vy*(j+1)*0.1+y_0
    data[str(traj[i,0])+"-"+str(traj[i,1])]=predict_array

save_path="test_file/test/phy_test.pkl"
with open(save_path,"wb") as f:
    pkl.dump(data,f)

#将所有的预测结果保存到csv文件中。
csv_dict_path="test_file/csv_list.pkl"
with open(csv_dict_path,"rb")as f:
    csv_dict=pkl.load(f)

phy_data_path="test_file/test/phy_test.pkl"
with open(phy_data_path,"rb")as f:
    phy_data=pkl.load(f)

net_data_path="test_file/test/net_test.pkl"
with open(net_data_path,"rb")as f:
    net_data=pkl.load(f)

import csv
import numpy as np
csvFile = open("csvData.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(["id","px","py","index"])
writer.writerows(np.array([[1,2,3,4],[5,6,7,8]]))
csvFile.close()

for csv_name in csv_dict:
    csvFile=open("csv_out/"+csv_name,"w")
    writer=csv.writer(csvFile)
    writer.writerow(["id", "px", "py", "index"])
    csvFile.close()

csv_dict_new={v:k for (k,v) in csv_dict.items()}

for (key,traj) in phy_data.items():

    csvid=float(key.split("-")[0])
    vehid=float(key.split("-")[1])

    csvFile = open("csv_out/" + csv_dict_new[csvid], "a")

    csv_out=np.full((50,4),0,dtype=float)
    csv_out[:,0]=vehid
    csv_out[:,1:3]=traj
    csv_out[:,3]=np.arange(1,51)*-1
    writer=csv.writer(csvFile)
    writer.writerows(csv_out)
    csvFile.close()

for (key,traj) in net_data.items():

    csvid=float(key.split("-")[0])
    vehid=float(key.split("-")[1])

    csvFile = open("csv_out/" + csv_dict_new[csvid], "a")

    csv_out=np.full((50,4),0,dtype=float)
    csv_out[:,0]=vehid
    csv_out[:,1:3]=traj
    csv_out[:,3]=np.arange(1,51)*-1
    writer=csv.writer(csvFile)
    writer.writerows(csv_out)
    csvFile.close()