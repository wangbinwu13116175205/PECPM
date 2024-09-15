import numpy as np
import os
import os.path as osp
import pdb
import networkx as nx
from utils.common_tools import mkdirs
import tqdm
import random

def z_score(arr):
    if len(arr.shape) >=3:
        col0_mins = np.min(arr[..., 0], axis=0, keepdims=True)
        col0_maxs = np.max(arr[..., 0], axis=0, keepdims=True)
        normalized_col0 = (arr[..., 0] - col0_mins) / (col0_maxs - col0_mins)
        normalized_arr = arr.copy()
        normalized_arr[...,0] = normalized_col0
        
        return normalized_arr
    else:
        return (data - np.mean(data)) / np.std(data)
def get_temporal_feature(data):
    
    data=np.expand_dims(data, axis=-1)
    feature_list = [data]
    n=data.shape[1]
    steps_per_day=288
    tod = [i % steps_per_day /steps_per_day for i in range(data.shape[0])]
    tod = np.array(tod)
    tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
    feature_list.append(tod_tiled)
    dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
    dow = np.array(dow)
    dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
    feature_list.append(dow_tiled)
    processed_data = np.concatenate(feature_list, axis=-1)
    return processed_data

def generate_dataset(data, idx, x_len=12, y_len=12,temporal_feature=False):
    res = data[idx]
    node_size = data.shape[1]
    C=data.shape[-1]
    t = len(idx)-1
    idic = 0
    x_index, y_index = [], []
    
    for i in tqdm.tqdm(range(t,0,-1)):
        if i-x_len-y_len>=0:
            x_index.extend(list(range(i-x_len-y_len, i-y_len)))
            y_index.extend(list(range(i-y_len, i)))

    x_index = np.asarray(x_index)
    y_index = np.asarray(y_index)
    if temporal_feature:
        x = res[x_index].reshape((-1, x_len, node_size,C))
        y = res[y_index].reshape((-1, y_len, node_size,C))
        x =x.transpose(0,2,1,3).reshape(-1,node_size,x_len*C)
        y =y.transpose(0,2,1,3)
        return x, y[...,0]  #L,N,T,C
    else:
        x = res[x_index].reshape((-1, x_len, node_size))
        y = res[y_index].reshape((-1, y_len, node_size))
        return x, y

def generate_samples122(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False,temporal_feature=False):
    edge_index = np.array(list(graph.edges)).T
    del graph
    data = data[0:days*288, :]
    t, n = data.shape[0], data.shape[1]
    if temporal_feature:     
        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
        

        train_x, train_y = generate_dataset(data, train_idx)
        val_x, val_y = generate_dataset(data, val_idx)
        test_x, test_y = generate_dataset(data, test_idx)
        if val_test_mix:
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)
        #np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
    else:
        data=get_temporal_feature(data)

        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
        
        train_x, train_y = generate_dataset(data, train_idx,temporal_feature)
        val_x, val_y = generate_dataset(data, val_idx,temporal_feature)
        test_x, test_y = generate_dataset(data, test_idx,temporal_feature)
        if val_test_mix:
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)
        #np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}      
    return data



class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False,temporal_feature=True):
    edge_index = np.array(list(graph.edges)).T
    del graph
    data = data[0:days*288, :]
    t, n = data.shape[0], data.shape[1]
    if temporal_feature==False:  
        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
        

        train_x, train_y = generate_dataset(data, train_idx)
        val_x, val_y = generate_dataset(data, val_idx)
        test_x, test_y = generate_dataset(data, test_idx)
        if val_test_mix:
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]
        
        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)
        #np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
    else:
        data=get_temporal_feature(data)
        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
        
        train_x, train_y = generate_dataset(data, train_idx,temporal_feature=temporal_feature)
        val_x, val_y = generate_dataset(data, val_idx,temporal_feature=temporal_feature)
        test_x, test_y = generate_dataset(data, test_idx,temporal_feature=temporal_feature)
        if val_test_mix:
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)
        #np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}      
    return data

def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False,temporal_feature=True):
    edge_index = np.array(list(graph.edges)).T
    del graph
    data = data[0:days*288, :]
    t, n = data.shape[0], data.shape[1]
    if temporal_feature==False:     
        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
        

        train_x, train_y = generate_dataset(data, train_idx)
        val_x, val_y = generate_dataset(data, val_idx)
        test_x, test_y = generate_dataset(data, test_idx)
        if val_test_mix:
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)
        #np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
    else:
        data=get_temporal_feature(data) #L,N,C

        train_idx = [i for i in range(int(t*train_rate))]
        val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
        test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]

        train_x, train_y = generate_dataset(data, train_idx,temporal_feature=temporal_feature)
        val_x, val_y = generate_dataset(data, val_idx,temporal_feature=temporal_feature)
        test_x, test_y = generate_dataset(data, test_idx,temporal_feature=temporal_feature)
        if val_test_mix:
            val_test_x = np.concatenate((val_x, test_x), 0)
            val_test_y = np.concatenate((val_y, test_y), 0)
            val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
            np.random.shuffle(val_test_idx)
            val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
            test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]
        train_x = z_score(train_x) #(5332, 655, 12, 3)
        val_x = z_score(val_x)
        test_x = z_score(test_x)

        #np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
        data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}      
    return data
def get_idx(days, savepath, data, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False,temporal_feature=True):
    del graph
    data = data[0:days*288, :]
    t, n = data.shape[0], data.shape[1]
     
    train_idx = [i for i in range(int(t*train_rate))]
    val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
    test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]     
    return train_idx,val_idx,test_idx
if __name__ == "__main__":
    for year in range(2011,2018):
        data_path=osp.join('/home/wbw/ijcai/data/district3F11T17/finaldata',str(year)+'.npz')
        data=np.load(data_path)['x']

        edge_path=osp.join('/home/wbw/ijcai/data/district3F11T17/FastData',str(year)+'_30day.npz')
        edge_index=np.load(edge_path)['edge_index']
        generate_samples1(data,edge_index=edge_index,year=year)