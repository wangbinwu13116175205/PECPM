import os
import shutil
import sys
import networkx as nx
import re
import json
import numpy as np
import math
import logging
import time
from datetime import datetime
from queue import Queue
from sklearn.preprocessing import MultiLabelBinarizer
import threading
import gc

import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp

class our_DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0

        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def graph_matrix_reader(file):
    df = pd.read_csv(file, header=None, index_col=None)
    return np.asarray(df.values)

def dict_add(d, key, add):
    if key in d:
        d[key] += add
    else:
        d[key] = add

def check_attr(params, attr, default):
    if attr not in params:
        params[attr] = default
        return False
    return True

def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

def load_fea(file_path):
    X = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            items = line.split()
            if len(items) < 1:
                continue
            X.append([float(item) for item in items])
    return np.array(X)


def symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        os.remove(dst)
        os.symlink(src, dst)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)

def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")

def append_to_file(file_path, s):
    with open(file_path, "a") as f:
        f.write(s)


# def mkdir(path):
#     """Judge whether the path exists and make dirs
#     :return: Boolean, if path exists then return True
#     """
#     if os.path.exists(path) == False:
#          os.makedirs(path)
#          return False
#     return True

def rmtree(path):
    if os.path.exists(path) == True:
        shutil.rmtree(path)
        return True
    return False

def get_logger(log_filename=None, module_name=__name__, level=logging.INFO):
    # select handler
    if log_filename is None:
        handler = logging.StreamHandler()
    elif type(log_filename) is str:
        handler = logging.FileHandler(log_filename, 'w')
    else:
        raise ValueError("log_filename invalid!")

    # build logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(('%(asctime)s %(filename)s' \
                    '[line:%(lineno)d] %(levelname)s %(message)s'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    return [i[1] for i in lst]

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        print((end_time - start_time).seconds)
        return res
    return wrapper



def module_decorator(func):
    def wrapper(*args, **kwargs):
        print("[+] Start %s ..." % (kwargs["mdl_name"], ))
        kwargs["info"]["log"].info("Start Module %s" % (kwargs["mdl_name"], ))
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        gc.collect()
        print("[+] Finished!\n[+] During Time: %.2f\n"  % (end_time - start_time).seconds)
        kwargs["info"]["log"].info(
                "[+] Finished!\n[+] During Time: %.2f\n" % (end_time - start_time).seconds)
        res["Duration"] = (end_time - start_time).seconds
        kwargs["info"]["log"].info("Module Results: " + str(res))
        print("[+] Module Results: " + str(res))
        kwargs["info"]["log"].info("[+] Module Results: " + str(res))
        return res
    return wrapper

def load_multilabel_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    lst = [i[1:] for i in lst]
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(lst)

def load_onehot_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    return np.array([i[1:] for i in lst], dtype=int)


import torch

def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse
