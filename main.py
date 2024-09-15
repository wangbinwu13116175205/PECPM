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
from utils.data_convert import generate_samples, get_idx
from src.model.mode24423 import Basic_Model
from src.model.ewc4 import EWC
from src.trafficDataset6 import TrafficDataset
from src.model import detect
from src.model import replay

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
    #if 'tcn2.weight' in state_dict:
    #    del state_dict['tcn2.weight']
    #    del state_dict['tcn2.bias']
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
def cosine_distance(A, B):
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)  # (m, 1)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)  # (p, 1)
    
    # 计算 A 和 B 的点积
    dot_product = np.dot(A, B.T)  # (m, p)
    
    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B.T)  # (m, p)
    
    return similarity

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
def match_attention(data,args):
    attention=cosine_distance(data,args.last_clusterc)
    return keep_top_k(attention,args.attention_weight[args.year-args.begin_year]) 

def train(inputs, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    if args.loss == "mse": lossfunc1 = func.mse_loss
    elif args.loss == "huber": lossfunc = func.smooth_l1_loss
    lossfunc= masked_mae
    cluster_lossf=masked_mse
    #train_idx,val_idx,test_idx= get_idx(inputs)
    # Dataset Definition
    N=inputs['train_x'].shape[1]
    pathatt='/home/wbw/ijcai5/TrafficStream-main/TrafficStream-main2/data_our/data/'+str(args.year)+'_attention.npy'
    attention=np.load(pathatt)
    C=attention.shape[-1]
    attention=attention.reshape(-1,N,C)

    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader =DataLoader(TrafficDataset("", "", x=inputs["train_x"][:, args.subgraph.numpy(),:], y=inputs["train_y"][:,  args.subgraph.numpy(),:],\
            att=attention[:, args.subgraph.numpy(),:],edge_index="", mode="subgraph"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset("", "", x=inputs["val_x"][:, args.subgraph.numpy(),:], y=inputs["val_y"][:, args.subgraph.numpy(),:], \
             att=attention[:, args.subgraph.numpy(),:],edge_index="", mode="subgraph"), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work) 
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        path = osp.join(args.path, str(args.year))
        #np.save(osp.join(path,'adj.npy'),adj)
        #np.save('/home/wbw/ijcai5/TrafficStream-main/TrafficStream-main2/results/'+str(args.year)+'.npy',adj)
        #vars(args)["attention"]=args.attention[ args.subgraph.numpy(),:]
    else:
        train_loader = DataLoader(TrafficDataset(inputs, "train",att=attention), batch_size=args.batch_size[args.year-args.begin_year], shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset(inputs, "val",att=attention), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test",att=attention), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        if args.ewc:
            #args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            model = EWC(gnn_model, args.adj, args.ewc_lambda[args.year-args.begin_year], args.ewc_strategy)
            ewc_loader = DataLoader(TrafficDataset(inputs, "train",att=attention), batch_size=args.batch_size[args.year-args.begin_year], shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            model.register_ewc_params(ewc_loader, lossfunc, device)
        else:
            model = gnn_model
    else:
        gnn_model = Basic_Model(args).to(args.device)
        model = gnn_model

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr[args.year-args.begin_year])
    #lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    args.logger.info("[*] Year " + str(args.year) + " Training start")
#    global_train_steps = len(train_loader) // args.batch_size +1

    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 10
    model.train()
    use_time = []

    for epoch in range(100):
        training_loss = 0.0
        start_time = datetime.now()
        
        # Train Model
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            pred,attention = model(data, args.sub_adj)#torch.Size([83840, 64])
            batch_att=pred.shape[0]//args.sub_adj.shape[0]
            loss_cluster=0
            
            #attention_label=match_attention(data.x.cpu(),args)

            #attention_label=torch.from_numpy(args.attention.repeat(batch_att,axis=0)).to(args.device)
            attention_label=data.att.to(args.device)
            loss_cluster = func.mse_loss(attention,attention_label)          
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]
            #loss = lossfunc(data.y, pred, reduction="mean")
            mask_value = torch.tensor(0.0)
            if data.y.min() < 1:
                mask_value = data.y.min()
            #print(loss_cluster)
            #print(lossfunc(data.y,pred, mask_value),loss_cluster)0.0036,
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

        # Early Stop
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

        #best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
        #best_model = Basic_Model(args)
        #best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
        #best_model = best_model.to(args.device)
            
            # Test Model
        #test_model2(model, args, test_loader, pin_memory)
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = Basic_Model(args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
            
            # Test Model
    test_model2(best_model, args, test_loader, pin_memory)
    result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))
        #args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))

def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            #pred, _ = to_dense_batch(pred, batch=data.batch)
            #data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred)   
            truth_.append(data)
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ =torch.cat(pred_, 0)
        truth_ = torch.cat(truth_, 0)
        mask_value = 0.0
        if truth_.min() < 1:
            mask_value = truth_.min()
        mae =all_metric(truth_, pred_, args,mask_value)
        return loss



def test_model2(model, args, testset, pin_memory):
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
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mask_value = torch.tensor(0)
        if truth_.min() < 1:
            mask_value = truth_.min()
        mae = metric(truth_, pred_, args,mask_value)
        return loss

def metric(ground_truth, prediction, args,mask_value):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
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
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)
        #path='/home/wbw/ijcai5/TrafficStream-main/TrafficStream-main2/data/'+str(year)+'_30day1.npz'
        #np.savez(path,train_x=inputs['train_x'],train_y=inputs['train_y'],val_x=inputs['val_x'],val_y=inputs['val_y'],test_x=inputs['test_x'],test_y=inputs['test_y'])
        #continue
        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year)))) 

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        if year == args.begin_year and args.load_first_year:
            # Skip the first year, model has been trained and retrain is not needed
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model2(model, args, test_loader, pin_memory=True)
            continue

        
        if year > args.begin_year and args.strategy == "incremental":
            # Load the best model
            model, _ = load_best_model(args)
            
            node_list = list()
            # Obtain increase nodes
            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            # Obtain influence nodes
            if args.detect:
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
                # 20% of current graph size will be sampled
                vars(args)["topk"] = int(0.01*args.graph_size) 
                influence_node_list = detect.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                node_list.extend(list(influence_node_list))

            # Obtain sample nodes
            if args.replay:
                vars(args)["replay_num_samples"] = int(0.09*args.graph_size) #int(0.2*args.graph_size)- len(node_list)
                args.logger.info("[*] replay node number {}".format(args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(args, inputs, model)
                node_list.extend(list(replay_node_list))
            
            node_list = list(set(node_list))
            if len(node_list) > int(0.2*args.graph_size):
                node_list = random.sample(node_list, int(0.15*args.graph_size))
            
            # Obtain subgraph of node list
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


        # Skip the year when no nodes needed to be trained incrementally
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