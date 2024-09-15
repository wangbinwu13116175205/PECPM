import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gcn_conv import BatchGCNConv


class Basic_Model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout[args.year-args.begin_year]
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.tcn2 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        self.memory=nn.Parameter(torch.FloatTensor(args.cluster,args.gcn["out_channel"]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.memory, gain=1.414)

        #self.time_emb = nn.Parameter(torch.empty(args.gcn["out_channel"]))
        #nn.init.xavier_uniform_(self.node_emb)

        self.args = args
        self.w1=nn.Linear(args.gcn["hidden_channel"],args.gcn["hidden_channel"])
        self.w2=nn.Linear(args.gcn["out_channel"],args.gcn["out_channel"])

        self.w3=nn.Linear(args.gcn["in_channel"]*2,args.gcn["hidden_channel"])

        
    def forward(self, data, adj):
        #B,T,N,C
        #data：class，x,y, data.x:bs*N,T
        T=self.args.x_len
        N = adj.shape[0]
        if len(data.x.shape)==4:
            res_x=data.x[...,0]
            x =data.x[...,0].reshape((-1,N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        else:
            input = data.x.reshape(-1,N,self.args.x_len,3)
            res_x=input[...,0].reshape(-1,self.args.x_len)
            x =input[...,0].reshape((-1,N, self.args.gcn["in_channel"]))   # [bs, N, feature]     
            tem_feature=input[...,1:3].reshape((-1,N, self.args.gcn["in_channel"]*2))
        #tem_x = self.w3(x[:,:,T:])        
        x = F.relu(self.gcn1(x, adj)+self.w3(tem_feature))                        # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature] torch.Size([83840, 1, 64])            
        x = self.tcn1(x)
        x = torch.mul((x), self.sigmoid(self.w1(x)))                                           # [bs * N, 1, feature] [83840, 1, 64]
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature] torch.Size([128, 655, 12])
        x = self.gcn2(x, adj)
        x = x.reshape((-1, 1, self.args.gcn["out_channel"]))                         #torch.Size([128, 655, 12])
        x=self.tcn2(x)       
 
        x = torch.mul((x), self.sigmoid(self.w2(x)))                                                    # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        attention = torch.matmul(x,self.memory.transpose(0,1)) #B*N, memory_size #B*N, memory_size                               
        attention=F.softmax(attention,dim=1)                         
        z=torch.matmul(attention,self.memory)
        x = x + res_x+z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x,attention


 
    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        print(data.x.shape)
        return x