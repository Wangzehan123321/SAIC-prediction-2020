# import numpy as np
# a=np.array([[1,2,3],[4,5,6],[3,4,1]])
# print(a[np.argsort(a[:,0]),:])
import pickle as pkl
import numpy as np
with open("../data/forecasting_features_val.pkl","rb")as f:
    data=pkl.load(f)
print(data["FEATURES"].values)

