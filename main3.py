import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from Bio.Cluster import kcluster,clustercentroids
from scipy.spatial.distance import cosine
from torch import optim
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np,masked_mae,masked_mape,masked_mse
from utils.data_convert import generate_samples
from src.model.model5 import Basic_Model
from src.model.ewc4 import EWC
from src.trafficDataset1 import TrafficDataset
from src.model import detect
from src.model import replay

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = True
n_work = 16 

def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year+1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))): 
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+args.time, str(args.year-1), loss[0]+".pkl")
        
    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    model = Basic_Model(args)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model, loss[0]

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True
def cosine_distance(matrix_a, matrix_c):
    a, b = matrix_a.shape
    c, _ = matrix_c.shape

    # 初始化注意力矩阵
    attention_matrix = np.zeros((a, c))

    # 计算注意力矩阵
    for i in range(a):
        for j in range(c):
            distance = cosine(matrix_a[i], matrix_c[j])
            attention_matrix[i, j] = 1 - distance
    return attention_matrix
def keep_top_k(matrix, k):
    # 对每行进行排序，返回排序后的索引
    sorted_indices = np.argsort(matrix, axis=1)
    
    # 生成一个与matrix形状相同的全零矩阵
    result = np.zeros_like(matrix)
    
    # 将前K个最大值设置为原始数值，其他设置为0
    rows = np.arange(matrix.shape[0])[:, np.newaxis]
    top_k_indices = sorted_indices[:, -k:]
    result[rows, top_k_indices] = matrix[rows, top_k_indices]
    
    return result
def long_term_pattern(inputs,args):
    # T,L,N
    inputs=inputs['train_x']

    T=inputs.shape[0]
    L=inputs.shape[1]
    N=inputs.shape[2]
    data=inputs[::12,:,:]
    days=data.shape[0]//24
    L=24*days
    data23=data[:L,:,:].reshape(-1,288,N)
    data23=data23.sum(axis=0)
    data24=data23.transpose(1,0)
    if args.year==args.begin_year:
        attention,_,_=kcluster(data24,nclusters=args.cluster,dist='u')
        clusterc,_=clustercentroids(data24,clusterid=attention)
        vars(args)["last_clusterc"] = clusterc
        attention=cosine_distance(data24,args.last_clusterc)
        np_attention=keep_top_k(attention,args.attention_weight[args.year-args.begin_year]) 
    else:
        attention,_,_=kcluster(data24,nclusters=args.cluster,dist='u')
        clusterc,_=clustercentroids(data24,clusterid=attention)
        vars(args)["last_clusterc"] = clusterc
        attention=cosine_distance(data24,args.last_clusterc)
        np_attention=keep_top_k(attention,args.attention_weight[args.year-args.begin_year]) 
 
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)
    #torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
    np.save(osp.join(path,'attention.npy'),np_attention.astype(np.float32))
    return np_attention.astype(np.float32)

def train(inputs, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    if args.loss == "mse": lossfunc1 = func.mse_loss
    elif args.loss == "huber": lossfunc = func.smooth_l1_loss
    lossfunc= masked_mae
    cluster_lossf=masked_mae
    # Dataset Definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(TrafficDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work) 
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        path = osp.join(args.path, str(args.year))
        np.save(osp.join(path,'adj.npy'),adj)
        vars(args)["attention"]=args.attention[ args.subgraph.numpy(),:]
    else:
        train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset(inputs, "val"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        if args.ewc:
            model = EWC(gnn_model, args.adj, args.ewc_lambda[args.year-args.begin_year], args.ewc_strategy)
            ewc_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            model.register_ewc_params(ewc_loader, lossfunc, device)
        else:
            model = gnn_model
    else:
        gnn_model = Basic_Model(args).to(args.device)
        model = gnn_model

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr[args.year-args.begin_year])

    args.logger.info("[*] Year " + str(args.year) + " Training start")


    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 10
    model.train()
    use_time = []
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(device, non_blocking=pin_memory)
            
            optimizer.zero_grad()
            pred,attention = model(data, args.sub_adj)
            batch_att=pred.shape[0]//args.sub_adj.shape[0]
            loss_cluster=0
            loss_cluster = cluster_lossf(attention,data.attention_label,torch.tensor(0.0))          
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]

            mask_value = torch.tensor(0)
            if data.y.min() < 1:
                mask_value = data.y.min()

            loss = lossfunc(data.y,pred, mask_value)+loss_cluster*args.beita[args.year-args.begin_year]
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()
            training_loss += float(loss)
            loss.backward()
            optimizer.step()
            
            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn 
 
        # Validate Model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device,non_blocking=pin_memory)
                pred,_ = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                mask_value = torch.tensor(0)
                if data.y.min() < 1:
                    mask_value = data.y.min()
                loss = lossfunc(data.y,pred, mask_value)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        test_model(model, args, val_loader, pin_memory,mode='val')
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = Basic_Model(args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
            

    test_model(best_model, args, test_loader, pin_memory,mode='test')
    result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}




def test_model(model, args, testset, pin_memory,mode='test'):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred,_ = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())   
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss/cn
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mask_value = torch.tensor(0)
        if truth_.min() < 1:
            mask_value = truth_.min()
        mae = metric(truth_, pred_, args,mask_value,mode)
        return loss

def metric(ground_truth, prediction, args,mask_value,mode):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
        #select hyperparameters
        if mode=='val' and i==3:
            print('--------------------------------',mode,'--------------------------------------')
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            print('--------------------------------',mode,'--------------------------------------')
    return mae

def all_metric(ground_truth, prediction, args,mask_value):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae(ground_truth[:, :, :i], prediction[:, :, :i], mask_value).item()
        rmse = masked_mse(ground_truth[:, :, :i], prediction[:, :, :i], mask_value).item()
        mape = masked_mape(ground_truth[:, :, :i], prediction[:, :, :i], mask_value).item()
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        # Load Data 
        graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        
        input_path=args.input_path+'/'+str(year)+'_inputs.npz'
        inputs = np.load(input_path)

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        attention=long_term_pattern(inputs,args)
        vars(args)["attention"]=attention
        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            continue

        
        if year > args.begin_year and args.strategy == "incremental":
            # Load the best model
            model, _ = load_best_model(args)
            
            node_list = list()

            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            if args.detect:
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
                
                replay_num=int(0.09*args.graph_size)
                evo_num=int(0.01*args.graph_size)
                replay_node,evo_node=detect.get_eveloved_nodes(args,replay_num,evo_num)
                
                node_list.extend(list(replay_node))
                node_list.extend(list(evo_node))
            
            node_list = list(set(node_list))
            if len(node_list) > int(0.2*args.graph_size):
                node_list = random.sample(node_list, int(0.2*args.graph_size))
            
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
                
           
            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)


        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        

        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
                test_model(model, args, test_loader, pin_memory=True)


    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            all12=0
            for year in range(args.begin_year, args.end_year+1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            all12=all12+result[i][j][year]
                            info+="{:.2f}\t".format(result[i][j][year])
            info+="{:.2f}\t".format(all12/7)
            logger.info("{}\t{}\t".format(i,j) + info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/test.json")
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--seed", type = int, default = 3208)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "/home/wbw/ijcai5/TrafficStream-main/TrafficStream-main2/res/district3F11T17/best212-2/2011/15.9668.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    seed_set(args.seed)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)
