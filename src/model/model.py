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
        T=self.args.x_len
        N = adj.shape[0]
        if len(data.x.shape)==4:
            res_x=data.x[...,0]
            x =data.x[...,0].reshape((-1,N, self.args.gcn["in_channel"])) 
        else:
            input = data.x.reshape(-1,N,self.args.x_len,3)
            res_x=input[...,0].reshape(-1,self.args.x_len)
            x =input[...,0].reshape((-1,N, self.args.gcn["in_channel"]))   
            tem_feature=input[...,1:3].reshape((-1,N, self.args.gcn["in_channel"]*2))
        #tem_x = self.w3(x[:,:,T:])        
        x = F.relu(self.gcn1(x, adj)+self.w3(tem_feature))                       
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))              
        x = self.tcn1(x)
        x = torch.mul((x), self.sigmoid(self.w1(x)))                                         
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    
        x = self.gcn2(x, adj)
        x = x.reshape((-1, 1, self.args.gcn["out_channel"]))                     
        x=self.tcn2(x)       
 
        x = torch.mul((x), self.sigmoid(self.w2(x)))                                                
        x = x.reshape((-1, self.args.gcn["out_channel"]))         
        attention = torch.matmul(x,self.memory.transpose(0,1))                        
        attention=F.softmax(attention,dim=1)                         
        z=torch.matmul(attention,self.memory)
        x = x + res_x+z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x,attention


 
    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  
        x = F.relu(self.gcn1(x, adj))                             
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    

        x = self.tcn1(x)                                          

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    
        x = self.gcn2(x, adj)                                      
        x = x.reshape((-1, self.args.gcn["out_channel"]))         
        
        x = x + data.x
        print(data.x.shape)
        return x
