import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch 
from src.trafficDataset import continue_learning_Dataset
from torch_geometric.data import Data, Batch, DataLoader
import torch
from scipy.spatial import distance
from scipy.stats import wasserstein_distance as WD
import os.path as osp
# scipy.stats.entropy(x, y) 


def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=3)
    # feature shape [T', feature_dim, N]
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)
        node_size = feature.size()[1]
        # print("before permute:", feature.size())
        feature = feature.permute(1,0,2)

        # [N, T', feature_dim]
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)
    

def score_func(pre_data, cur_data, args):
    # shape: [T, N]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    # return staiton_id of topk max score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
def sort_with_index(lst):
    sorted_index = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return sorted_index
def random_sampling(data_size, num_samples):
    return np.random.choice(data_size, num_samples)
def get_eveloved_nodes(args,replay_num,evo_num):
    # should be N*T
    past_path=args.daily_node+'/'+str(args.year-1)+'.npy'
    daily_node_past=np.load(past_path)
    cuettern_path=args.daily_node+'/'+str(args.year)+'.npy'
    daily_node_cur=np.load(cuettern_path)
    if daily_node_past.shape[0]<daily_node_past.shape[1]:
        daily_node_cur=daily_node_cur.transpose(1,0)
        daily_node_past=daily_node_past.transpose(1,0)

    daily_node_cur=daily_node_cur[:daily_node_past.shape[0],:]

    distance=[]
    for i in range(daily_node_past.shape[0]):
        distance.append(WD(daily_node_past[i],daily_node_cur[i]))
    sorted_index = sort_with_index(distance)
    replay_node=sorted_index[-int(replay_num*0.1):]
    replay_list.extend(replay_node)
    evo_node=list(sorted_index[:evo_num])
    replay_sample=random_sampling(daily_node_past.shape[0],int(replay_num*0.9))
    replay_list.extend(replay_sample)
    return replay_list,evo_node
