import pickle as pkl
import torch
from SAIC_model import SAICNet,init_weights
from SAIC_utils import SAICDataset,maskedRMSE,get_xy_from_nt_seq
from torch.utils.data import DataLoader
# import time

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

save_pkl=True

# Initialize network
net = SAICNet(args)
net.load_state_dict(torch.load('./trained_models/SAIC_on_argoverse_pretrained_net/saicnet_840.tar'))

if args['use_cuda']:
    net = net.cuda()

tsSet = SAICDataset('test_file/net_data.pkl',val_metric=True,save_pkl=save_pkl)
tsDataloader = DataLoader(tsSet,batch_size=1024,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(50).cuda()#预测50个点，点间隔为0.1s
counts = torch.zeros(50).cuda()

save_data_dict={}
with torch.no_grad():
    for i, data in enumerate(tsDataloader):
        # st_time = time.time()
        hist,fut,op_mask,current_refer,oracle_ct,csv_batch,vehicle_batch = data

        # Initialize Variables
        if args['use_cuda']:
            hist = hist.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            current_refer=current_refer.cuda()

        if args['use_maneuvers']:
            #TODO:多模态实现
            pass
        else:
            fut_pred = net(hist)
            fut_pred = fut_pred.permute(1, 0, 2)
            fut_pred[:,:,1]=fut_pred[:,:,1]+current_refer
            fut_pred_abs=get_xy_from_nt_seq(fut_pred.cpu().detach().numpy(),oracle_ct)
            fut_pred_abs=torch.from_numpy(fut_pred_abs).permute(1, 0, 2)
            if save_pkl:
                for ii in range(len(csv_batch)):
                    save_data_dict[str(csv_batch[ii])+"-"+str(vehicle_batch[ii])]=fut_pred_abs[:,ii,:]
                save_path="test_file/test/net_test.pkl"
                with open(save_path,"wb")as f:
                    pkl.dump(save_data_dict,f)
